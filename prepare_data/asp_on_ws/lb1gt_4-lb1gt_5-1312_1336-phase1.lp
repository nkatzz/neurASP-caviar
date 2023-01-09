
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
orientation(lb1gt_4,90,1325). coords(lb1gt_4,310,112,1325).  visible(lb1gt_4,1325). happensAt(inactive(lb1gt_4),1325). orientation(lb1gt_5,133,1325). coords(lb1gt_5,287,97,1325).  visible(lb1gt_5,1325). happensAt(walking(lb1gt_5),1325).
orientation(lb1gt_4,90,1326). coords(lb1gt_4,310,112,1326).  visible(lb1gt_4,1326). happensAt(inactive(lb1gt_4),1326). orientation(lb1gt_5,133,1326). coords(lb1gt_5,288,97,1326).  visible(lb1gt_5,1326). happensAt(walking(lb1gt_5),1326).
orientation(lb1gt_4,90,1327). coords(lb1gt_4,310,112,1327).  visible(lb1gt_4,1327). happensAt(inactive(lb1gt_4),1327). orientation(lb1gt_5,133,1327). coords(lb1gt_5,289,97,1327).  visible(lb1gt_5,1327). happensAt(walking(lb1gt_5),1327).
orientation(lb1gt_4,90,1328). coords(lb1gt_4,310,112,1328).  visible(lb1gt_4,1328). happensAt(inactive(lb1gt_4),1328). orientation(lb1gt_5,133,1328). coords(lb1gt_5,290,97,1328).  visible(lb1gt_5,1328). happensAt(walking(lb1gt_5),1328).
orientation(lb1gt_4,90,1329). coords(lb1gt_4,310,112,1329).  visible(lb1gt_4,1329). happensAt(inactive(lb1gt_4),1329). orientation(lb1gt_5,133,1329). coords(lb1gt_5,291,97,1329).  visible(lb1gt_5,1329). happensAt(walking(lb1gt_5),1329).
orientation(lb1gt_4,90,1330). coords(lb1gt_4,310,112,1330).  visible(lb1gt_4,1330). happensAt(inactive(lb1gt_4),1330). orientation(lb1gt_5,133,1330). coords(lb1gt_5,291,98,1330).  visible(lb1gt_5,1330). happensAt(walking(lb1gt_5),1330).
orientation(lb1gt_4,90,1331). coords(lb1gt_4,310,112,1331).  visible(lb1gt_4,1331). happensAt(inactive(lb1gt_4),1331). orientation(lb1gt_5,133,1331). coords(lb1gt_5,291,98,1331).  visible(lb1gt_5,1331). happensAt(walking(lb1gt_5),1331).
orientation(lb1gt_4,90,1332). coords(lb1gt_4,310,112,1332).  visible(lb1gt_4,1332). happensAt(inactive(lb1gt_4),1332). orientation(lb1gt_5,133,1332). coords(lb1gt_5,292,98,1332).  visible(lb1gt_5,1332). happensAt(walking(lb1gt_5),1332).
orientation(lb1gt_4,90,1333). coords(lb1gt_4,310,112,1333).  visible(lb1gt_4,1333). happensAt(inactive(lb1gt_4),1333). orientation(lb1gt_5,133,1333). coords(lb1gt_5,292,98,1333).  visible(lb1gt_5,1333). happensAt(walking(lb1gt_5),1333).
orientation(lb1gt_4,90,1334). coords(lb1gt_4,310,112,1334).  visible(lb1gt_4,1334). happensAt(inactive(lb1gt_4),1334). orientation(lb1gt_5,128,1334). coords(lb1gt_5,292,98,1334).  visible(lb1gt_5,1334). happensAt(walking(lb1gt_5),1334).
orientation(lb1gt_4,90,1335). coords(lb1gt_4,310,112,1335).  visible(lb1gt_4,1335). happensAt(inactive(lb1gt_4),1335). orientation(lb1gt_5,128,1335). coords(lb1gt_5,292,98,1335).  visible(lb1gt_5,1335). happensAt(walking(lb1gt_5),1335).
orientation(lb1gt_4,90,1336). coords(lb1gt_4,310,112,1336).  visible(lb1gt_4,1336). happensAt(inactive(lb1gt_4),1336). orientation(lb1gt_5,129,1336). coords(lb1gt_5,292,98,1336).  visible(lb1gt_5,1336). happensAt(walking(lb1gt_5),1336).
#show holdsAt/2.