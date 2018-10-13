# Get images

X = []
for filename in os.listdir('../Train/'):
    X.append(img_to_array(load_img('../Train/'+filename)))
X = np.array(X, dtype=float)