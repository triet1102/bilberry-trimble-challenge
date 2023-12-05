# # Sample image score and file path
# image_score = 0.85
# image_paths = [
#     "/Users/triettran/bilberry-trimble-challenge/dataset/test_images/2.jpeg",
#     "/Users/triettran/bilberry-trimble-challenge/dataset/test_images/5.jpeg",
#     "/Users/triettran/bilberry-trimble-challenge/dataset/test_images/8.jpeg",
# ]

# image_scores = [0.85, 0.92, 0.78]

# # Create subplots with one row and the number of columns equal to the number of images
# fig = sp.make_subplots(
#     rows=1,
#     cols=len(image_paths),
#     subplot_titles=[f"Score: {score}" for score in image_scores],
# )


# # Add images and annotations to each subplot
# for i, (image_path, score) in enumerate(zip(image_paths, image_scores)):
#     image = Image.open(image_path)
#     fig.add_trace(
#         go.Image(
#             z=image,
#         ),
#         row=1,
#         col=i + 1,
#     )

# # Save the figure as an HTML file
# output_file_path = "output_plot.html"
# pio.write_html(fig, output_file_path, auto_open=False)

# print(f"Plots and Image Scores saved to: {output_file_path}")


from omegaconf import OmegaConf

config = OmegaConf.load("src/model/cnn/test_config.yaml")
print(type(config.model.param1))
print(type(config.model.param2))
print(config)
