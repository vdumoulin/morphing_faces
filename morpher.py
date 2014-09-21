import numpy

numpy.random.seed(112387)


class Morpher(object):
    """
    Generates images of faces using a variational autoencoder (VAE) to map
    a point Z in a 400-dimensional latent space to X, a 48 x 48 pixels image.

    The model was trained on the Toronto Face Database (TFD) and features two
    hidden layers with 2000 rectified linear units each. The prior distribution
    p(X) was chosen to be gaussian, and the conditional distribution p(X | Z)
    was also chosen to be gaussian.

    While the latent space is 400-dimensional, most of the dimensions in Z are
    ignored by the decoder as a result of the training procedure. Those that
    carry information (29 of them) can be used to vary the coordinates of Z.
    """
    # Mapping between the 29 useful dimensions in latent space and their index
    index_mapping = {0: 31, 1: 37, 2: 42, 3: 47, 4: 69, 5: 83, 6: 84, 7: 94,
                     8: 101, 9: 137, 10: 138, 11: 139, 12: 165, 13: 178,
                     14: 192, 15: 197, 16: 206, 17: 212, 18: 219, 19: 235,
                     20: 259, 21: 275, 22: 280, 23: 288, 24: 319, 25: 340,
                     26: 354, 27: 369, 28: 380}
    # Shape of the generated images
    image_shape = (48, 48)

    def __init__(self):
        # If true, ignore requests to change the coordinates of Z
        self.freeze_coordinates = False
        # Dimensions along which coordinates of Z will vary
        self.dimension_0 = self.index_mapping[28]
        self.dimension_1 = self.index_mapping[11]
        # Point in latent space, selected at random to begin with
        self.Z = self._sample_Z()

        # Model parameters
        self.d_W_2 = numpy.load('data/d_W_2.npy')
        self.d_b_2 = numpy.load('data/d_b_2.npy')
        self.d_W_1 = numpy.load('data/d_W_1.npy')
        self.d_b_1 = numpy.load('data/d_b_1.npy')
        self.d_W_0 = numpy.load('data/d_W_0.npy')
        self.d_b_0 = numpy.load('data/d_b_0.npy')

        self.e_W_2 = numpy.load('data/e_W_2.npy')
        self.e_b_2 = numpy.load('data/e_b_2.npy')
        self.e_W_1 = numpy.load('data/e_W_1.npy')
        self.e_b_1 = numpy.load('data/e_b_1.npy')
        self.e_W_0 = numpy.load('data/e_W_0.npy')
        self.e_b_0 = numpy.load('data/e_b_0.npy')

    def toggle_freeze(self):
        """
        Toggles on and off the option to ignore requests to change coordinates
        of Z
        """
        self.freeze_coordinates = not self.freeze_coordinates

    def _sample_Z(self):
        """
        Samples a point in latent space according to the prior distribution
        p(Z)
        """
        return numpy.random.normal(size=(400, ))

    def shuffle(self):
        """
        Resamples a new Z
        """
        self.Z = self._sample_Z()

    def select_dimensions(self, d_0, d_1):
        """
        Selects the two dimensions in Z along which coordinates will vary
        """
        if d_0 not in self.index_mapping or d_1 not in self.index_mapping:
            raise KeyError()
        self.dimension_0 = self.index_mapping[d_0]
        self.dimension_1 = self.index_mapping[d_1]

    def get_Z(self):
        """
        Returns the current point in latent space
        """
        return self.Z.copy()

    def set_Z(self, Z):
        """
        Sets the current point in latent space
        """
        self.Z = Z.copy()

    def set_coordinates(self, x, y):
        """
        Sets new coordinates for the two selected dimensions of Z if
        coordinates are not frozen
        """
        if not self.freeze_coordinates:
            self.Z[self.dimension_0] = x
            self.Z[self.dimension_1] = y

    def generate_face(self):
        """
        Maps the point Z in latent space to a 48 x 48 pixels face image
        """
        A_2 = numpy.dot(self.Z, self.d_W_2) + self.d_b_2
        H_2 = numpy.where(A_2 > 0.0, A_2, 0.0 * A_2)

        A_1 = numpy.dot(H_2, self.d_W_1) + self.d_b_1
        H_1 = numpy.where(A_1 > 0.0, A_1, 0.0 * A_1)

        A_0 = numpy.dot(H_1, self.d_W_0) + self.d_b_0
        X = 1.0 / (1.0 + numpy.exp(-A_0))

        return X.reshape(self.image_shape)

    def infer_Z(self, X):
        """
        Maps an 48 x 48 face image to its representation Z in latent space

        Parameters
        ----------
        X : numpy.array
            Array of shape (2304, ) or (48, 48) representing the face image
        """
        A_1 = numpy.dot(X, self.e_W_0) + self.e_b_0
        H_1 = numpy.where(A_1 > 0.0, A_1, 0.0 * A_1)

        A_2 = numpy.dot(H_1, self.e_W_1) + self.d_b_1
        H_2 = numpy.where(A_2 > 0.0, A_2, 0.0 * A_2)

        Z = numpy.dot(H_2, self.e_W_2) + self.e_b_2

        return Z


class MorpherFromModel(object):
    """
    Generates images of faces using a variational autoencoder (VAE) to map
    a point Z in a 400-dimensional latent space to X, a 48 x 48 pixels image.

    The model was trained on the Toronto Face Database (TFD) and features two
    hidden layers with 2000 rectified linear units each. The prior distribution
    p(X) was chosen to be gaussian, and the conditional distribution p(X | Z)
    was also chosen to be gaussian.

    While the latent space is 400-dimensional, most of the dimensions in Z are
    ignored by the decoder as a result of the training procedure. Those that
    carry information (29 of them) can be used to vary the coordinates of Z.
    """
    def __init__(self, model_path):
        self.model = serial.load(model_path)
        dataset_yaml_src = self.model.dataset_yaml_src
        dataset = yaml_parse.load(dataset_yaml_src)
        numpy_X = as_floatX(dataset.get_design_matrix())[:10000]

        X = T.matrix('X')
        phi = self.model.encode_phi(X)
        mu, log_sigma = phi
        kl_divergence_term = self.model.per_component_kl_divergence_term(phi)
        f = theano.function(inputs=[X], outputs=[mu, kl_divergence_term])
        numpy_mu, numpy_kl = f(numpy_X)

        kl_mean = numpy_kl.mean(axis=0).mean(axis=0)
        kl_std = numpy_kl.mean(axis=0).std(axis=0)
        self.valid_indexes = numpy.argwhere(
            numpy_kl.mean(axis=0) > kl_mean + kl_std
        )[:, 0]
        self.index_mapping = dict(
            [(i, j) for i, j in
             zip(self.valid_indexes, xrange(self.valid_indexes.shape[0]))]
        )

        image_size = int(numpy.sqrt(self.model.nvis))
        assert image_size ** 2 == self.model.nvis
        self.image_shape = (image_size, image_size)

        mu_mean = numpy_mu.mean(axis=0)
        mu_std = numpy_mu.std(axis=0)
        self.bounds = numpy.vstack([mu_mean - mu_std,
                                    mu_mean + mu_std])[:, self.valid_indexes]

        symbolic_Z = theano.tensor.matrix('Z')
        self._sample_Z_fn = theano.function(
            inputs=[],
            outputs=self.model.sample_from_p_z(1).flatten()
        )
        self._generate_face_fn = theano.function(
            inputs=[symbolic_Z],
            outputs=self.model.sample_from_p_x_given_z(
                1,
                self.model.decode_theta(symbolic_Z)
            ).flatten()
        )

        # If true, ignore requests to change the coordinates of Z
        self.freeze_coordinates = False
        # Dimensions along which coordinates of Z will vary
        self.dimension_0 = 0
        self.dimension_1 = 1
        # Point in latent space, selected at random to begin with
        self.Z = self._sample_Z()

    def toggle_freeze(self):
        """
        Toggles on and off the option to ignore requests to change coordinates
        of Z
        """
        self.freeze_coordinates = not self.freeze_coordinates

    def _sample_Z(self):
        """
        Samples a point in latent space according to the prior distribution
        p(Z)
        """
        return self._sample_Z_fn()

    def shuffle(self):
        """
        Resamples a new Z
        """
        self.Z = self._sample_Z()

    def select_dimensions(self, d_0, d_1):
        """
        Selects the two dimensions in Z along which coordinates will vary
        """
        if d_0 not in self.index_mapping or d_1 not in self.index_mapping:
            raise KeyError()
        self.dimension_0 = d_0
        self.dimension_1 = d_1

    def get_Z(self):
        """
        Returns the current point in latent space
        """
        return self.Z.copy()

    def set_Z(self, Z):
        """
        Sets the current point in latent space
        """
        self.Z = Z.copy()

    def set_coordinates(self, x, y):
        """
        Sets new coordinates for the two selected dimensions of Z if
        coordinates are not frozen
        """
        assert x >= 0 and x <= 1
        assert y >= 0 and y <= 1
        if not self.freeze_coordinates:
            x_min = self.bounds[0, self.dimension_0]
            x_max = self.bounds[1, self.dimension_0]
            self.Z[self.index_mapping[self.dimension_0]] = \
                x_min + x * (x_max - x_min)

            y_min = self.bounds[0, self.dimension_1]
            y_max = self.bounds[1, self.dimension_1]
            self.Z[self.index_mappint[self.dimension_1]] = \
                y_min + y * (y_max - y_min)

    def generate_face(self):
        """
        Maps the point Z in latent space to a 48 x 48 pixels face image
        """
        return self._generate_face_fn(self.Z[None, :]).reshape(self.image_shape)

    def infer_Z(self, X):
        """
        Maps an 48 x 48 face image to its representation Z in latent space

        Parameters
        ----------
        X : numpy.array
            Array of shape (2304, ) or (48, 48) representing the face image
        """
        return self._infer_Z_fn(X)
