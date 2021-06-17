from similarity_judgement import SimilarityJudgementModel

if __name__ == '__main__':
    SimilarityJudgementModel(
        train_image_path=r'.',
        validation_image_path=r'.',
        input_shape=(32, 32, 1),
        lr=0.01,
        epochs=500,
        batch_size=32).fit()
