// Mapping between the 29 useful dimensions in latent space and their index
var index_mapping = {
    0: 31, 1: 37, 2: 42, 3: 47, 4: 69, 5: 83, 6: 84, 7: 94, 8: 101, 9: 137,
    10: 138, 11: 139, 12: 165, 13: 178, 14: 192, 15: 197, 16: 206, 17: 212,
    18: 219, 19: 235, 20: 259, 21: 275, 22: 280, 23: 288, 24: 319, 25: 340,
    26: 354, 27: 369, 28: 380
};

// Shape of the generated images
var image_shape = [48, 48];

/*-----------------------------------------------------------------------------
MORPHER : Generates images of faces using a variational autoencoder (VAE) to map
          a point Z in latent space to an image X.

          The model was trained on the Toronto Face Database (TFD). The prior
          distribution p(X) was chosen to be gaussian, and the conditional
          distribution p(X | Z) was also chosen to be gaussian.

          While the latent space is high-dimensional, most of the dimensions in
          Z are ignored by the decoder as a result of the training procedure.
          Those that carry information can be used to vary the coordinates of Z.
-----------------------------------------------------------------------------*/
function Morpher () {
    // Number of parameters loaded
    this._params_loaded = 0;
    // Point in latent space, selected at random to begin with
    this.Z = this.sample_Z();
    // Model parameters
    var sizes = numeric.parseCSV(
        numeric.getURL("data/sizes.csv").responseText.slice(0,-1)
    )[0];
    this.image_size = Math.sqrt(sizes[0]);
    this.weights = [];
    this.biases = [];
    for(var i=0; i<sizes.length - 1; i++) {
        morpher = this;
        this.weights.push(
            load("data/d_W_" + i + ".bin", sizes[i] * sizes[i + 1],
                 sizes[i + 1], sizes[i], function(){morpher._params_loaded += 1;})
        );
        this.biases.push(
            load("data/d_b_" + i + ".bin", sizes[i], 1, sizes[i],
                 function(){morpher._params_loaded += 1;})
        );
    }
}

/*-----------------------------------------------------------------------------
MORPHER.sample_Z : Samples a point in latent space by sampling from a uniform
                   distribution over [-1, 1]^N
-----------------------------------------------------------------------------*/
Morpher.prototype.sample_Z = function() {
    return numeric.sub(numeric.mul(2, numeric.random([1, 400])), 1);
};

/*-----------------------------------------------------------------------------
MORPHER.shuffle : Resamples a new Z
-----------------------------------------------------------------------------*/
Morpher.prototype.shuffle = function() {
    this.Z = this.sample_Z();
};

/*-----------------------------------------------------------------------------
MORPHER.get_Z : Returns the current point in latent space
-----------------------------------------------------------------------------*/
Morpher.prototype.get_Z = function() {
    return this.Z;
};

/*-----------------------------------------------------------------------------
MORPHER.set_Z : Sets the current point in latent space
-----------------------------------------------------------------------------*/
Morpher.prototype.set_Z = function(Z) {
    this.Z = Z;
};

/*-----------------------------------------------------------------------------
MORPHER.set_coordinates : Sets new coordinates for the two selected dimensions
                          of Z if coordinates are not frozen
-----------------------------------------------------------------------------*/
Morpher.prototype.set_coordinate = function(i, z) {
    this.Z[0][index_mapping[i]] = z;
};

/*-----------------------------------------------------------------------------
MORPHER.generate_face : Maps the point Z in latent space to a 48 x 48 pixels
                        face image
-----------------------------------------------------------------------------*/
Morpher.prototype.generate_face = function() {
    // Latent to  last hidden mapping
    var A = numeric.add(
        numeric.dot(this.Z, this.weights[this.weights.length - 1]),
        this.biases[this.biases.length - 1]
    );
    var H = numeric.mul(A, numeric.geq(A, 0));
    // Hidden to hidden mapping
    for(var i=this.weights.length-2; i>0; i--) {
        A = numeric.add(numeric.dot(H, this.weights[i]), this.biases[i]);
        H = numeric.mul(A, numeric.geq(A, 0));
    }
    // Hidden to visible mapping
    A = numeric.add(numeric.dot(H, this.weights[0]), this.biases[0]);
    var flat_X = numeric.div(1.0, numeric.add(1.0, numeric.exp(numeric.neg(A))));
    var X = [];
    while(flat_X[0].length) X.push(flat_X[0].splice(0, this.image_size));
    return X;
};

Morpher.prototype.ready = function() {
    return this._params_loaded == 6;
};
