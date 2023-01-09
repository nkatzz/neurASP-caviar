
holdsAt(F,Te) :- initiatedAt(F,Ts), fluent(F), next(Ts,Te).
holdsAt(F,Te) :- holdsAt(F,Ts), not terminatedAt(F,Ts), fluent(F), next(Ts,Te).


fluent(moving(X,Y)) :- person(X), person(Y), X != Y.
fluent(meeting(X,Y)) :- person(X), person(Y), X != Y.


#include "../next.lp".

% Distance/orientation calculations:
#include "../distances.lp".

% Definitions of "type" predicates (extracted from the data):
time(T) :- happensAt(disappear(_),T).
time(T) :- happensAt(appear(_),T).
time(T) :- happensAt(active(_),T).
time(T) :- happensAt(inactive(_),T).
time(T) :- happensAt(walking(_),T).
time(T) :- happensAt(running(_),T).
time(T) :- coords(_,_,_,T).
time(T) :- orientation(_,_,T).


person(X) :- happensAt(disappear(X),_).
person(X) :- happensAt(appear(X),_).
person(X) :- happensAt(active(X),_).
person(X) :- happensAt(inactive(X),_).
person(X) :- happensAt(walking(X),_).
person(X) :- happensAt(running(X),_).
person(X) :- coords(X,_,_,_).
person(X) :- orientation(X,_,_).


% Definition for the "moving" complex event: 
initiatedAt(moving(X0,X1),X2) :- happensAt(walking(X0),X2),happensAt(walking(X1),X2),orientationMove(X0,X1,X2),close(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(active(X1),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(inactive(X1),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X1),X2),happensAt(inactive(X0),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(running(X0),X2), person(X1).
terminatedAt(moving(X0,X1),X2) :- happensAt(running(X1),X2), person(X0).
terminatedAt(moving(X0,X1),X2) :- happensAt(disappear(X0),X2), person(X1).
terminatedAt(moving(X0,X1),X2) :- happensAt(disappear(X1),X2), person(X0).

% Definition for the "meeting" complex event:
initiatedAt(meeting(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(active(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(inactive(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(inactive(X0),X2),happensAt(active(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(inactive(X0),X2),happensAt(inactive(X1),X2),close(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(running(X0),X2),person(X1).
terminatedAt(meeting(X0,X1),X2) :- happensAt(running(X1),X2),person(X0).
terminatedAt(meeting(X0,X1),X2) :- happensAt(disappear(X0),X2), person(X1).
terminatedAt(meeting(X0,X1),X2) :- happensAt(disappear(X1),X2), person(X0).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,25,X2).

	orientation(lb1gt_4,90,1300). coords(lb1gt_4,310,112,1300).  visible(lb1gt_4,1300). happensAt(inactive(lb1gt_4),1300). orientation(lb1gt_5,133,1300). coords(lb1gt_5,231,99,1300).  visible(lb1gt_5,1300). happensAt(walking(lb1gt_5),1300).
orientation(lb1gt_4,90,1301). coords(lb1gt_4,310,112,1301).  visible(lb1gt_4,1301). happensAt(inactive(lb1gt_4),1301). orientation(lb1gt_5,133,1301). coords(lb1gt_5,235,98,1301).  visible(lb1gt_5,1301). happensAt(walking(lb1gt_5),1301).
orientation(lb1gt_4,90,1302). coords(lb1gt_4,310,112,1302).  visible(lb1gt_4,1302). happensAt(inactive(lb1gt_4),1302). orientation(lb1gt_5,133,1302). coords(lb1gt_5,239,98,1302).  visible(lb1gt_5,1302). happensAt(walking(lb1gt_5),1302).
orientation(lb1gt_4,90,1303). coords(lb1gt_4,310,112,1303).  visible(lb1gt_4,1303). happensAt(inactive(lb1gt_4),1303). orientation(lb1gt_5,133,1303). coords(lb1gt_5,241,97,1303).  visible(lb1gt_5,1303). happensAt(walking(lb1gt_5),1303).
orientation(lb1gt_4,90,1304). coords(lb1gt_4,310,112,1304).  visible(lb1gt_4,1304). happensAt(inactive(lb1gt_4),1304). orientation(lb1gt_5,133,1304). coords(lb1gt_5,244,97,1304).  visible(lb1gt_5,1304). happensAt(walking(lb1gt_5),1304).
orientation(lb1gt_4,90,1305). coords(lb1gt_4,310,112,1305).  visible(lb1gt_4,1305). happensAt(inactive(lb1gt_4),1305). orientation(lb1gt_5,133,1305). coords(lb1gt_5,246,96,1305).  visible(lb1gt_5,1305). happensAt(walking(lb1gt_5),1305).
orientation(lb1gt_4,90,1306). coords(lb1gt_4,310,112,1306).  visible(lb1gt_4,1306). happensAt(inactive(lb1gt_4),1306). orientation(lb1gt_5,133,1306). coords(lb1gt_5,248,96,1306).  visible(lb1gt_5,1306). happensAt(walking(lb1gt_5),1306).
orientation(lb1gt_4,90,1307). coords(lb1gt_4,310,112,1307).  visible(lb1gt_4,1307). happensAt(inactive(lb1gt_4),1307). orientation(lb1gt_5,133,1307). coords(lb1gt_5,250,95,1307).  visible(lb1gt_5,1307). happensAt(walking(lb1gt_5),1307).
orientation(lb1gt_4,90,1308). coords(lb1gt_4,310,112,1308).  visible(lb1gt_4,1308). happensAt(inactive(lb1gt_4),1308). orientation(lb1gt_5,133,1308). coords(lb1gt_5,251,95,1308).  visible(lb1gt_5,1308). happensAt(walking(lb1gt_5),1308).
orientation(lb1gt_4,90,1309). coords(lb1gt_4,310,112,1309).  visible(lb1gt_4,1309). happensAt(inactive(lb1gt_4),1309). orientation(lb1gt_5,133,1309). coords(lb1gt_5,252,94,1309).  visible(lb1gt_5,1309). happensAt(walking(lb1gt_5),1309).
orientation(lb1gt_4,90,1310). coords(lb1gt_4,310,112,1310).  visible(lb1gt_4,1310). happensAt(inactive(lb1gt_4),1310). orientation(lb1gt_5,133,1310). coords(lb1gt_5,253,94,1310).  visible(lb1gt_5,1310). happensAt(walking(lb1gt_5),1310).
orientation(lb1gt_4,90,1311). coords(lb1gt_4,310,112,1311).  visible(lb1gt_4,1311). happensAt(inactive(lb1gt_4),1311). orientation(lb1gt_5,133,1311). coords(lb1gt_5,254,94,1311).  visible(lb1gt_5,1311). happensAt(walking(lb1gt_5),1311).
orientation(lb1gt_4,90,1312). coords(lb1gt_4,310,112,1312).  visible(lb1gt_4,1312). happensAt(inactive(lb1gt_4),1312). orientation(lb1gt_5,133,1312). coords(lb1gt_5,256,93,1312).  visible(lb1gt_5,1312). happensAt(walking(lb1gt_5),1312).
orientation(lb1gt_4,90,1313). coords(lb1gt_4,310,112,1313).  visible(lb1gt_4,1313). happensAt(inactive(lb1gt_4),1313). orientation(lb1gt_5,133,1313). coords(lb1gt_5,261,93,1313).  visible(lb1gt_5,1313). happensAt(walking(lb1gt_5),1313).
orientation(lb1gt_4,90,1314). coords(lb1gt_4,310,112,1314).  visible(lb1gt_4,1314). happensAt(inactive(lb1gt_4),1314). orientation(lb1gt_5,133,1314). coords(lb1gt_5,265,93,1314).  visible(lb1gt_5,1314). happensAt(walking(lb1gt_5),1314).
orientation(lb1gt_4,90,1315). coords(lb1gt_4,310,112,1315).  visible(lb1gt_4,1315). happensAt(inactive(lb1gt_4),1315). orientation(lb1gt_5,133,1315). coords(lb1gt_5,269,93,1315).  visible(lb1gt_5,1315). happensAt(walking(lb1gt_5),1315).
orientation(lb1gt_4,90,1316). coords(lb1gt_4,310,112,1316).  visible(lb1gt_4,1316). happensAt(inactive(lb1gt_4),1316). orientation(lb1gt_5,133,1316). coords(lb1gt_5,272,94,1316).  visible(lb1gt_5,1316). happensAt(walking(lb1gt_5),1316).
orientation(lb1gt_4,90,1317). coords(lb1gt_4,310,112,1317).  visible(lb1gt_4,1317). happensAt(inactive(lb1gt_4),1317). orientation(lb1gt_5,133,1317). coords(lb1gt_5,275,94,1317).  visible(lb1gt_5,1317). happensAt(walking(lb1gt_5),1317).
orientation(lb1gt_4,90,1318). coords(lb1gt_4,310,112,1318).  visible(lb1gt_4,1318). happensAt(inactive(lb1gt_4),1318). orientation(lb1gt_5,133,1318). coords(lb1gt_5,277,94,1318).  visible(lb1gt_5,1318). happensAt(walking(lb1gt_5),1318).
orientation(lb1gt_4,90,1319). coords(lb1gt_4,310,112,1319).  visible(lb1gt_4,1319). happensAt(inactive(lb1gt_4),1319). orientation(lb1gt_5,133,1319). coords(lb1gt_5,280,94,1319).  visible(lb1gt_5,1319). happensAt(walking(lb1gt_5),1319).
orientation(lb1gt_4,90,1320). coords(lb1gt_4,310,112,1320).  visible(lb1gt_4,1320). happensAt(inactive(lb1gt_4),1320). orientation(lb1gt_5,133,1320). coords(lb1gt_5,281,95,1320).  visible(lb1gt_5,1320). happensAt(walking(lb1gt_5),1320).
orientation(lb1gt_4,90,1321). coords(lb1gt_4,310,112,1321).  visible(lb1gt_4,1321). happensAt(inactive(lb1gt_4),1321). orientation(lb1gt_5,133,1321). coords(lb1gt_5,282,95,1321).  visible(lb1gt_5,1321). happensAt(walking(lb1gt_5),1321).
orientation(lb1gt_4,90,1322). coords(lb1gt_4,310,112,1322).  visible(lb1gt_4,1322). happensAt(inactive(lb1gt_4),1322). orientation(lb1gt_5,133,1322). coords(lb1gt_5,283,95,1322).  visible(lb1gt_5,1322). happensAt(walking(lb1gt_5),1322).
orientation(lb1gt_4,90,1323). coords(lb1gt_4,310,112,1323).  visible(lb1gt_4,1323). happensAt(inactive(lb1gt_4),1323). orientation(lb1gt_5,133,1323). coords(lb1gt_5,283,96,1323).  visible(lb1gt_5,1323). happensAt(walking(lb1gt_5),1323).
orientation(lb1gt_4,90,1324). coords(lb1gt_4,310,112,1324).  visible(lb1gt_4,1324). happensAt(inactive(lb1gt_4),1324). orientation(lb1gt_5,133,1324). coords(lb1gt_5,285,96,1324).  visible(lb1gt_5,1324). happensAt(walking(lb1gt_5),1324).
#show holdsAt/2.