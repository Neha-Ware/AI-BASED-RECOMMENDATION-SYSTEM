import org.apache.mahout.cf.taste.eval.RecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.eval.AverageAbsoluteDifferenceRecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.recommender.GenericItemBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;

import java.io.File;
import java.util.List;

public class RecommendationSystem {
    public static void main(String[] args) throws Exception {
        // Load data
        DataModel model = new FileDataModel(new File("data.csv"));

        // Define item similarity
        ItemSimilarity similarity = new PearsonCorrelationSimilarity(model);

        // to Create recommender
        GenericItemBasedRecommender recommender = new GenericItemBasedRecommender(model, similarity);

        // this is Recommend items for a user
        List<RecommendedItem> recommendations = recommender.recommend(1, 5);

        for (RecommendedItem recommendation : recommendations) {
            System.out.println(recommendation);
        }

        // to Evaluate recommender
        RecommenderEvaluator evaluator = new AverageAbsoluteDifferenceRecommenderEvaluator();
        double output = evaluator.evaluate(recommender, null, model, 0.7, 1.0);
        System.out.println("Evaluation result: " + output);
    }
}
