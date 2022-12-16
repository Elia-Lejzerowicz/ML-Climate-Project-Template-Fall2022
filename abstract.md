Today, 3.7% of total carbon emissions are attributable to the Internet and its supporting technologies, which is equivalent to the entire airline industry. Additionally, it is anticipated that this number will double over the following three years. Change and impact are required. This study's objective is to discover a means of lowering the amount of carbon dioxide released by websites. In fact, the more complex a website is, the more energy it uses to load and carbon emissions it produces. Developers must be made aware of the best practices to use in order to lessen the carbon footprint of their websites.

Utilizing Google Lighthouse and GreenIT-analysis, two open-source audit tools, we analyzed more than 500 websites to create a dataset. The size of the DOM, the quantity of requests, and the size of the page are the three metrics used by Greenit to calculate a score (formula: (https://blog.octo.com/wp-content/uploads/2021/01/1-5-ecoindex-formule-recol ored.jpg)). Then, Google Lighthouse offers a score based on six metrics, including the time to interactive, the speed index, etc. These Chrome extensions, along with additional data preprocessing, produced a score from 0-100 that evaluated the extent to which the 50 best practices recommended, which make up our dataset, were implemented. We're interested in learning which elements have the biggest effects on the Eco-Index score because it can be difficult for websites to implement all of the audit tools' recommendations to raise the Eco-index score
.
Therefore, the first goal of this study is to determine which of the 50 practices has the greatest influence on the EcoIndex score. Four machine learning regression tasks are carried out, and two different models—Random Forest and XGboost—are used and assessed for each of them in order to identify the feature importance that corresponds to the most crucial practices to implement. 

To determine the effect of implementing only those recommendations, we created a new dataset by applying the most crucial practices (manually raising relevant scores), and we predicted the new eco-index score.

This study aims to assist website developers by offering straightforward and practical eco-concept recommendations. It is also an extension of the EcoIndex project (GreenIt.fr) and a current contribution to the Accenture Technology EcoSonar open-source project (ecosonar.org), which aims to create an application that makes specific recommendations for websites.



Metrics:

The score of the EcoIndex depends on the following metrics:

1. The Size of the DOM
2. Number of Requests
3. Size of the page

The score of Google Lighthouse Performance depends on the following metrics:

4. First Contentful Paint
5. Speed Index
6. Largest Contentful Paint
7. Time to Interactive
8. Total Blocking Time
9. Cumulative Layout Shift

Features:

List of Features from Green-IT Analysis :

1. cache header ratio
2. compress ratio
3. domains number
4. images resized in browser number
5. empty src tag number
6. inline style sheets number
7. inline js scripts number
8. error number
9. js validate
10. umber of requests
11. images downloaded not displayed number
12. max cookies length
13. percent minified css
14. percent minified js
15. total cookies size
16. redirect number
17. total min gains (bitmap images)
18. total size to optimize (svg images)
19. plugins number
20. print style sheets number
21. number social network button
22. style sheets number
23. eTags Ratio
24. total Fonts Size


List of Features from Google Ligthhouse:

25. viewport
26. serverResponsetime
27. mainthreadWorkBreakdown
28. bootupTime
29. fontDisplay
30. thirdPartySummary
31. thirdPartyFacades
32. lcpLazyLoaded
33. nonCompositedAnimations
34. domSize
35. usesLongCacheTtl
36. usesResponsiveImages
37. offscreenImages
38. unusedCssRules
39. unusedJavscript
40. usesOptimizedImages
41. modernImageFormats
42. usesTextCompression
43. usesHttp2
44. efficientAnimatedContent
45. legacyJavascript
46. totalByteWeight
47. noDocumentWrite
48. layoutShiftElements
49. usesPassiveEventListeners



