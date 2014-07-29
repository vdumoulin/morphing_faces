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
          a point Z in a 400-dimensional latent space to X, a 48 x 48 pixels
          image.

          The model was trained on the Toronto Face Database (TFD) and features
          two hidden layers with 2000 rectified linear units each. The prior
          distribution p(X) was chosen to be gaussian, and the conditional
          distribution p(X | Z) was also chosen to be gaussian.

          While the latent space is 400-dimensional, most of the dimensions in Z
          are ignored by the decoder as a result of the training procedure.
          Those that carry information (29 of them) can be used to vary the
          coordinates of Z.
-----------------------------------------------------------------------------*/
function Morpher () {
    // Number of parameters loaded
    this._params_loaded = 0;
    // If true, ignore requests to change the coordinates of Z
    this.freeze_coordinates = false;
    // Dimensions along which coordinates of Z will vary
    this.dimension_0 = index_mapping[28];
    this.dimension_1 = index_mapping[11];
    // Point in latent space, selected at random to begin with
    this.Z = this.sample_Z();
    // Model parameters
    this.d_W_2 = this._load("data/d_W_2.bin", 400 * 2000, 2000, 400);
    this.d_W_1 = this._load("data/d_W_1.bin", 2000 * 2000, 2000, 2000);
    this.d_W_0 = this._load("data/d_W_0.bin", 48 * 48 * 2000, 48 * 48, 2000);
    this.d_b_2 = this._load("data/d_b_2.bin", 2000, 0, 2000);
    this.d_b_1 = this._load("data/d_b_1.bin", 2000, 0, 2000);
    this.d_b_0 = this._load("data/d_b_0.bin", 48 * 48, 0, 48 * 48);

    // self.e_W_2 = numpy.load('data/e_W_2.npy')
    // self.e_b_2 = numpy.load('data/e_b_2.npy')
    // self.e_W_1 = numpy.load('data/e_W_1.npy')
    // self.e_b_1 = numpy.load('data/e_b_1.npy')
    // self.e_W_0 = numpy.load('data/e_W_0.npy')
    // self.e_b_0 = numpy.load('data/e_b_0.npy')
}

Morpher.prototype.get_d_b_2 = function() {
    return this.d_b_2;
}

/*-----------------------------------------------------------------------------
MORPHER.toggle_freeze : Toggles on and off the option to ignore requests to
                        change coordinates of Z
-----------------------------------------------------------------------------*/
Morpher.prototype.toggle_freeze = function() {
    this.freeze_coordinates = !this.freeze_coordinates;
};

/*-----------------------------------------------------------------------------
MORPHER.sample_Z : Samples a point in latent space according to the prior
                   distribution p(Z)
-----------------------------------------------------------------------------*/
Morpher.prototype.sample_Z = function() {
    return numeric.sub(numeric.mul(1, numeric.random([400])), 1);
};

/*-----------------------------------------------------------------------------
MORPHER.shuffle : Resamples a new Z
-----------------------------------------------------------------------------*/
Morpher.prototype.shuffle = function() {
    this.Z = this.sample_Z();
};

/*-----------------------------------------------------------------------------
MORPHER.select_dimensions : Selects the two dimensions in Z along which
                            coordinates will vary
-----------------------------------------------------------------------------*/
Morpher.prototype.select_dimensions = function(d_0, d_1) {
    // TODO: error handling
    this.dimension_0 = index_mapping[d_0];
    this.dimension_1 = index_mapping[d_1];
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
Morpher.prototype.set_coordinates = function(x, y) {
    if(!this.freeze_coordinates) {
        this.Z[this.dimension_0] = x;
        this.Z[this.dimension_1] = y;
    }
};

Morpher.prototype.set_one_coordinate = function(i, z) {
    if(!this.freeze_coordinates) {
        this.Z[index_mapping[i]] = z;
    }
};

/*-----------------------------------------------------------------------------
MORPHER.generate_face : Maps the point Z in latent space to a 48 x 48 pixels
                        face image
-----------------------------------------------------------------------------*/
Morpher.prototype.generate_face = function() {
    var A_2 = numeric.add(numeric.dot(this.d_W_2, this.Z), this.d_b_2);
    var H_2 = numeric.mul(A_2, numeric.geq(A_2, 0));

    var A_1 = numeric.add(numeric.dot(this.d_W_1, H_2), this.d_b_1);
    var H_1 = numeric.mul(A_1, numeric.geq(A_1, 0));

    var A_0 = numeric.add(numeric.dot(this.d_W_0, H_1), this.d_b_0);
    var flat_X = numeric.div(
        1.0, numeric.add(1.0, numeric.exp(numeric.neg(A_0)))
    );
    var X = [];
    while(flat_X.length) X.push(flat_X.splice(0, 48));
    return X;
};

/*-----------------------------------------------------------------------------
MORPHER.infer_Z : Maps an 48 x 48 face image to its representation Z in latent
                  space
-----------------------------------------------------------------------------*/
Morpher.prototype.infer_Z = function(X) {
    // A_1 = numpy.dot(X, self.e_W_0) + self.e_b_0
    // H_1 = numpy.where(A_1 > 0.0, A_1, 0.0 * A_1)

    // A_2 = numpy.dot(H_1, self.e_W_1) + self.d_b_1
    // H_2 = numpy.where(A_2 > 0.0, A_2, 0.0 * A_2)

    // Z = numpy.dot(H_2, self.e_W_2) + self.e_b_2

    // return Z
};

Morpher.prototype.ready = function() {
    return this._params_loaded == 6;
};

/*-----------------------------------------------------------------------------
MORPHER._load : Loads a parameter array by file name
-----------------------------------------------------------------------------*/
Morpher.prototype._load = function(filename, length, M, N) {
    var array = [];
    var oReq = new XMLHttpRequest();
    oReq.open("GET", filename, true);
    oReq.responseType = "arraybuffer";

    var morpher = this;

    oReq.onload = function (oEvent) {
        var arrayBuffer = oReq.response; // Note: not oReq.responseText
        if (arrayBuffer) {
            var float32Array = new Float32Array(arrayBuffer);
            if(M != 0){
                for (var i=0; i<M; i++) {
                    row = [];
                    for(var j=0; j<N; j++) {
                        row.push(float32Array[(M * j) + i]);
                    }
                    array.push(row);
                }
            }
            else {
                for(var j=0; j<N; j++) {
                    array.push(float32Array[j]);
                }
            }
            console.log('Done loading ' + filename);
            morpher._params_loaded += 1 ;
        }
    };

    oReq.send(null);

    return array;
};
