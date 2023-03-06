using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.DependencyInjection;
using Newtonsoft.Json;
using OpenAI.Api;
using TensorFlow;

namespace MyWebApp
{
    public class Program
    {
        static Dictionary<string, float[]> embeddings;
        static List<string[]> trainingData;
        static List<string[]> qaPairs;
        static IChatbotAPI chatbot;

        static async Task Main(string[] args)
        {
            var result = await LoadEmbeddingsAsync("embeddings.csv");
            embeddings = result.Item1;
            trainingData = result.Item2;
            qaPairs = new List<string[]>();
            chatbot = new MyChatbotAPI("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");

            var host = new WebHostBuilder()
                .UseKestrel()
                .ConfigureServices(services =>
                {
                    services.AddRouting();
                })
                .Configure(app =>
                {
                    app.UseRouting();

                    app.UseEndpoints(endpoints =>
                    {
                        endpoints.MapGet("/", async context =>
                        {
                            await context.Response.WriteAsync(
                                "Welcome to the one and only ROM Bot IQ." +
                                "Please enter your question in the 'question' parameter.");
                        });

                        endpoints.MapGet("/answer", async context =>
                        {
                            string question = context.Request.Query["question"].ToString();
                            if (string.IsNullOrWhiteSpace(question))
                            {
                                context.Response.StatusCode = (int)HttpStatusCode.BadRequest;
                                await context.Response.WriteAsync("Please enter a question.");
                                return;
                            }

                            try
                            {
                                string answer = await GetAnswerAsync(question);
                                qaPairs.Add(new string[] { question, answer });
                                await context.Response.WriteAsync(answer);
                            }
                            catch (Exception ex)
                            {
                                context.Response.StatusCode = (int)HttpStatusCode.InternalServerError;
                                await context.Response.WriteAsync("An error occurred while processing your question. Please try again later.");
                            }
                        });

                        endpoints.MapGet("/train", async context =>
                        {
                            await TrainModelAsync();
                            await context.Response.WriteAsync("Training completed successfully.");
                        });
                    });
                })
                .Build();

            host.Run();
        }

        static async Task<Dictionary<string, float[]>> LoadEmbeddingsAsync(string filePath)
        {
            using (var reader = new StreamReader(filePath))
            {
                var embeddings = await reader.ReadToEndAsync();

                return embeddings
                    .Split(new[] { "\r\n", "\r", "\n" }, StringSplitOptions.RemoveEmptyEntries)
                    .Skip(1)
                    .Select(line => line.Split(','))
                    .ToDictionary(
                        fields => fields[0],
                        fields => fields.Skip(1).Select(f => float.Parse(f)).ToArray()
                    );
            }
        }

        static async Task<List<string[]>> LoadTrainingDataAsync(string filePath)
        {
            using (var reader = new StreamReader(filePath))
            {
                var data = await reader.ReadToEndAsync();

                return data
                    .Split(new[] { "\r\n", "\r", "\n" }, StringSplitOptions.RemoveEmptyEntries)
                    .Skip(1)
                    .Select(line => line.Split(','))
                    .ToList();
            }
        }

        static async Task<string> GetAnswerAsync(string question)
        {
            string closestEmbedding = GetClosestEmbedding(question);

            if (closestEmbedding != null)
            {
                return closestEmbedding;
            }
            else
            {
                string answer = await chatbot.GetAnswerAsync(question);
                qaPairs.Add(new string[] { question, answer });
             float[] embedding = await GenerateEmbeddingAsync(answer);
             embeddings[answer] = embedding;
             return answer;
         }
     }

     static async Task<float[]> GenerateEmbeddingAsync(string text)
     {
         OpenAI openAI = new OpenAI("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");
         Response response = await openAI.Engines.ComputeAsync(text);

         return response.Embedding;
     }

     static string GetClosestEmbedding(string question)
     {
         var matches = qaPairs.Where(qa => string
             .Equals(qa[0], StringComparison.OrdinalIgnoreCase));

         if (matches.Any())
         {
             return matches.OrderByDescending(qa => CosineSimilarity(
                 embeddings[qaPairs.IndexOf(qa)[1]],
                 embeddings[qaPairs.Count][1])).First()[1];
         }
         else
         {
             return null;
         }
     }

     static float CosineSimilarity(float[] embedding1, float[] embedding2)
     {
         float dotProduct = 0.0f;

         for (int i = 0; i < embedding1.Length; i++)
         {
             dotProduct += embedding1[i] * embedding2[i];
         }

         float magnitude1 = (float)Math.Sqrt(embedding1.Sum(x => Math.Pow(x, 2)));
         float magnitude2 = (float)Math.Sqrt(embedding2.Sum(x => Math.Pow(x, 2)));

         float similarity = dotProduct / (magnitude1 * magnitude2);

         return similarity;
     }

     static async Task TrainModelAsync()
     {
         // Preprocess the training data
         List<string> questions = new List<string>();
         List<string> answers = new List<string>();
         foreach (var qa in qaPairs)
         {
             questions.Add(qa[0]);
             answers.Add(qa[1]);
         }

         // Train a text classification model using TensorFlow
         var tf = new TFInterface();
         var model = tf.LoadModel("classification.txt");

         var input = tf.Placeholder(TFDataType.String);
         var output = tf.Placeholder(TFDataType.Float);
         var vocab = tf.LoadVocabulary("vocabulary.txt");

         var preprocess = tf.PreprocessText(input, vocab, 256, false);
         var predict = tf.PredictTextClassification(model, preprocess, 2, output);

         var loss = tf.SoftmaxCrossEntropyWithLogits(output, preprocess.Labels);
         var train = tf.TrainAdamOptimizer(loss, 0.001f).Minimize(loss);

         using (var session = tf.Session())
         {
             session.InitializeVariables();

             for (int epoch = 1; epoch <= 10; epoch++)
             {
                 for (int i = 0; i < questions.Count; i++)
                 {
                     var question = questions[i];
                     var answer = answers[i];

                     var label = (answer == GetClosestEmbedding(question)) ? 1 : 0;

                     var feed = new Dictionary<TFOutput, TFTensor>
                     {
                         [input] = question.ToTensor(),
                         [output] = label.ToTensor()
                     };

                     session.Run(train, feed);
                 }
             }

             // Save the trained model
             tf.SaveModel(model, "classification.txt");
            }
        }
    }
}