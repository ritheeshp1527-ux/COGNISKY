from transformers import pipeline

emotion_model = pipeline(
    "text-classification",
    model="joeddav/distilbert-base-uncased-go-emotions-student",
    return_all_scores=True
)

out = emotion_model("hello world")
print(out)
