from sentiment_analysis import SentimentAnalysis

sa = SentimentAnalysis()
sentiment = sa.analyze_sentiment("This is best film i seen")
print(sa.model.layers[1].output)
print(sentiment)

intermediate_out = sa.debug_hidden_layer_out("This is best film i seen")

sentiment = sa.analyze_sentiment("Wonderful but suck")
print(sentiment)


sentiment = sa.analyze_sentiment("Ha ha i laught")
print(sentiment)