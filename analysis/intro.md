# Introduction      

The earth is an amazing planet that cultivates branches of animals. In general, scholars split them into 12 classes including mammals, birds, reptiles, amphibians, fishes, insects, crustaceans, arachnids, echinoderms, worms, mollusks and sponges {cite:p}`types-of-animals`. The traditional way in animal classification is manually identifying the characteristics and attributing it the mostly close class {cite:p}`animal-classification`. However, it is tedious and time consuming, especially when the data set is very huge. A question hereby comes to us, if we can apply K-nearest neighbors (KNN) algorithms in predicting the type an animal belongs to given its related characteristics, such as hair, feathers, etc.? Therefore, in this project, we will show how we use KNN to do classification in animals based on data set {cite:p}`zoo-data` which contains 1 categorical attribute, 17 Boolean-valued attributes and 1 numerical attribute. The categorical attribute appears to be the class attribute. Detailed breakdowns are as follows:  

1. `animal name`: Unique for each instance        
2. `hair`: Boolean        
3. `feathers`: Boolean        
4. `eggs`: Boolean        
5. `milk`: Boolean        
6. `airborne`: Boolean        
7. `aquatic`: Boolean        
8. `predator`: Boolean        
9. `toothed`: Boolean        
10. `backbone`: Boolean        
11. `breathes`: Boolean        
12. `venomous`: Boolean        
13. `fins`: Boolean        
14. `legs`: Numeric (set of values: {0,2,4,5,6,8})        
15. `tail`: Boolean        
16. `domestic`: Boolean        
17. `catsize`: Boolean        
18. `type`: Numeric (integer values in range [1,7])        
