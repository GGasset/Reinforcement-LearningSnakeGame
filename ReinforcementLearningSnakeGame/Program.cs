using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeatNetwork;
using NeatNetwork.Libraries;
using SnakeGame;
using System.IO;

namespace ReinforcementLearningSnakeGame
{
    internal class Program
    {
        static void Main(string[] args)
        {
            var gridSize = 50;
            var learningRate = .5;
            var game = new SnakeGame.SnakeGame(gridSize);
            var n = new ReinforcementLearningNN(new NN(new int[] { game.GetBoardInfo().Length, 50, 35, 20, 40, 10, 1 }, Activation.ActivationFunctions.Sigmoid), learningRate);

            Console.WriteLine("Set-up completed!");
            Console.Beep();
            var saveInsteadOfHumanInteraction = GetUserInput("Want to save the Networks or be supervising the networks (a sound will be played when ready to be interacted with) 0/1 - ") == "0";
            var gamesBetweenHumanInteraction = saveInsteadOfHumanInteraction? Convert.ToInt32(GetUserInput("Games between saves - ")) : Convert.ToInt32(GetUserInput("Games between human interaction - "));

            string folderPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            if (!folderPath.EndsWith(@"\"))
                folderPath += @"\";
            string folderName = @"Snake Reinforcement learning\";

            string fileName = "network.txt";
            string path = folderPath + folderName;

            bool directoryExists = Directory.Exists(path);
            if (!directoryExists)
                Directory.CreateDirectory(path);

            folderPath = path;
            
            path += fileName;

            string[] files;
            bool multipleFilesInFolder = (files = Directory.GetFiles(folderPath)).Length > 1, fileExists = false;
            if (!multipleFilesInFolder)
            {
                fileExists = File.Exists(path);
            }

            if (fileExists || multipleFilesInFolder)
            {
                var response = GetUserInput("Read Network from disk? Y/N (the program will delete all the network files but you can save them again later) - ");
                var readFile = response == "y";
                if (readFile)
                {
                    n = new ReinforcementLearningNN(new NN(DiskHandler.ReadFromDisk(multipleFilesInFolder? files[files.Length - 1].Remove(0, files[files.Length-1].LastIndexOf(@"\") + 1) : fileName, folderPath)), learningRate);
                }
                //Directory.Delete(folderPath, true);
                Directory.CreateDirectory(folderPath);
            }

            int trainingSessionsCounter = 0;
            int gameCounter = 0;
            while (true)
            {
                for (int i = 0; i < gamesBetweenHumanInteraction; i++)
                {
                    game = new SnakeGame.SnakeGame(gridSize);
                    double lastEatenFood = 0;
                    double lastDistanceToFood = game.FoodDistance;
                    double maxDistance = game.MaxBoardDistance;
                    bool ateInPreviousRound = false;
                    double minDistanceToFood = game.MaxBoardDistance;
                    while (!game.isFinished)
                    {
                        SnakeGame.SnakeGame.Direction direction;
                        game.Move(direction = SigmoidToDirection(n.Execute(game.GetBoardInfo())[0]));
                        bool foodHasBeenEaten = lastEatenFood != game.EatenFood;

                        double foodDistance = game.FoodDistance;
                        double reward = -n.Reward + ((maxDistance - foodDistance) - (maxDistance - lastDistanceToFood)) * 3;
                        reward += reward * 3 * Convert.ToInt16(reward < 0);
                        reward += Math.Pow(reward, 3) * Convert.ToInt16(foodHasBeenEaten);
                        n.GiveReward(reward);
                        lastEatenFood = game.EatenFood;
                        lastDistanceToFood = game.FoodDistance;
                        if (ateInPreviousRound)
                            n.GiveReward(-n.Reward);

                        ateInPreviousRound = foodHasBeenEaten;
                        minDistanceToFood -= (minDistanceToFood - game.FoodDistance) * Convert.ToInt16(minDistanceToFood < foodDistance);

                        if (false)
                        {
                            Console.WriteLine("------------------------------");
                            Console.WriteLine(game.GetBoardString());
                        }
                    }
                    n.GiveReward(-100);
                    n.TerminateAgent();
                    Console.WriteLine($"Snake lived {gameCounter} lives, it ate {game.EatenFood} apples and it was this close to food: {minDistanceToFood}!");
                    gameCounter++;
                }

                if (saveInsteadOfHumanInteraction)
                {
                    DiskHandler.SaveToDisk(fileName.Replace(".txt", $"{trainingSessionsCounter}.txt"), folderPath, n.n.ToString());
                }
                else
                {
                    Console.Beep();
                    var showGame = GetUserInput("Want to take a look to a currentGame? Y/N - ") == "y";
                    if (showGame)
                        ShowGame(GetGame(n.n, gridSize));

                }
                Console.WriteLine($"Training session {trainingSessionsCounter} finished!");
                trainingSessionsCounter++;
            }
        }

        private static string GetUserInput(string prompt, bool makeAnswerLowercase = true)
        {
            Console.Write(prompt);
            var response = Console.ReadLine();
            if (makeAnswerLowercase)
                response = response.ToLower();

            return response;
        }

        private static SnakeGame.SnakeGame.Direction SigmoidToDirection(double networkOutput)
        {
            if (networkOutput < -.5)
                return SnakeGame.SnakeGame.Direction.Down;
            else if (networkOutput <= 0)
                return SnakeGame.SnakeGame.Direction.Right;
            else if (networkOutput >= .5)
                return SnakeGame.SnakeGame.Direction.Left;
            else
                return SnakeGame.SnakeGame.Direction.Up;
        }

        private static void ShowGame(string[] strs, int msBetweenUpdates = 250)
        {
            foreach (var str in strs)
            {
                Console.WriteLine(str);
                System.Threading.Thread.Sleep(msBetweenUpdates);
            }
        }

        private static string[] GetGame(NN n, int gridSize)
        {
            SnakeGame.SnakeGame game = new SnakeGame.SnakeGame(gridSize);
            List<string> output = new List<string>();
            while (!game.isFinished)
            {
                output.Add(game.GetBoardString());
                game.Move(SigmoidToDirection(n.Execute(game.GetBoardInfo())[0]));
            }
            return output.ToArray();
        }
    }
}
