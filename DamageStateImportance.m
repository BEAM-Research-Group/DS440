

Features = table(Elevation, Stories, Latitude, Longitude, Roof_Type, Building_Type,Random_Column,Distance_From_Shoreline);


%For the regression trees

t = templateTree('NumVariablesToSample','all',...
    'PredictorSelection','interaction-curvature','Surrogate','off');

%Need 'Surrogate', 'on' for missing data

rng(1);
Model = fitrensemble(Features,DamageState,'Method','Bag','NumLearningCycles',200, ...
    'Learners',t);

impOOB = oobPermutedPredictorImportance(Model);

figure
bar(impOOB)
title('Unbiased Predictor Importance Estimates for Hurricane Michael Data')
xlabel('Predictor Variable')
ylabel('Feature Importance')
h = gca;
h.XTickLabel = Model.PredictorNames;
h.XTickLabelRotation = 45;
h.TickLabelInterpreter = 'none';