
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
orientation(lb1gt_4,90,1337). coords(lb1gt_4,310,112,1337).  visible(lb1gt_4,1337). happensAt(inactive(lb1gt_4),1337). orientation(lb1gt_5,128,1337). coords(lb1gt_5,293,98,1337).  visible(lb1gt_5,1337). happensAt(walking(lb1gt_5),1337).
orientation(lb1gt_4,103,1338). coords(lb1gt_4,310,112,1338).  visible(lb1gt_4,1338). happensAt(inactive(lb1gt_4),1338). orientation(lb1gt_5,127,1338). coords(lb1gt_5,293,98,1338).  visible(lb1gt_5,1338). happensAt(walking(lb1gt_5),1338).
orientation(lb1gt_4,101,1339). coords(lb1gt_4,310,112,1339).  visible(lb1gt_4,1339). happensAt(inactive(lb1gt_4),1339). orientation(lb1gt_5,127,1339). coords(lb1gt_5,293,98,1339).  visible(lb1gt_5,1339). happensAt(walking(lb1gt_5),1339).
orientation(lb1gt_4,104,1340). coords(lb1gt_4,310,112,1340).  visible(lb1gt_4,1340). happensAt(inactive(lb1gt_4),1340). orientation(lb1gt_5,130,1340). coords(lb1gt_5,293,98,1340).  visible(lb1gt_5,1340). happensAt(walking(lb1gt_5),1340).
orientation(lb1gt_4,112,1341). coords(lb1gt_4,309,113,1341).  visible(lb1gt_4,1341). happensAt(inactive(lb1gt_4),1341). orientation(lb1gt_5,122,1341). coords(lb1gt_5,292,98,1341).  visible(lb1gt_5,1341). happensAt(walking(lb1gt_5),1341).
orientation(lb1gt_4,118,1342). coords(lb1gt_4,309,113,1342).  visible(lb1gt_4,1342). happensAt(inactive(lb1gt_4),1342). orientation(lb1gt_5,119,1342). coords(lb1gt_5,292,98,1342).  visible(lb1gt_5,1342). happensAt(walking(lb1gt_5),1342).
orientation(lb1gt_4,118,1343). coords(lb1gt_4,309,113,1343).  visible(lb1gt_4,1343). happensAt(inactive(lb1gt_4),1343). orientation(lb1gt_5,119,1343). coords(lb1gt_5,291,99,1343).  visible(lb1gt_5,1343). happensAt(walking(lb1gt_5),1343).
orientation(lb1gt_4,118,1344). coords(lb1gt_4,309,114,1344).  visible(lb1gt_4,1344). happensAt(inactive(lb1gt_4),1344). orientation(lb1gt_5,112,1344). coords(lb1gt_5,291,101,1344).  visible(lb1gt_5,1344). happensAt(walking(lb1gt_5),1344).
orientation(lb1gt_4,118,1345). coords(lb1gt_4,308,114,1345).  visible(lb1gt_4,1345). happensAt(inactive(lb1gt_4),1345). orientation(lb1gt_5,112,1345). coords(lb1gt_5,291,104,1345).  visible(lb1gt_5,1345). happensAt(walking(lb1gt_5),1345).
orientation(lb1gt_4,118,1346). coords(lb1gt_4,308,114,1346).  visible(lb1gt_4,1346). happensAt(inactive(lb1gt_4),1346). orientation(lb1gt_5,112,1346). coords(lb1gt_5,291,107,1346).  visible(lb1gt_5,1346). happensAt(walking(lb1gt_5),1346).
orientation(lb1gt_4,118,1347). coords(lb1gt_4,308,114,1347).  visible(lb1gt_4,1347). happensAt(inactive(lb1gt_4),1347). orientation(lb1gt_5,112,1347). coords(lb1gt_5,291,109,1347).  visible(lb1gt_5,1347). happensAt(walking(lb1gt_5),1347).
orientation(lb1gt_4,123,1348). coords(lb1gt_4,307,113,1348).  visible(lb1gt_4,1348). happensAt(inactive(lb1gt_4),1348). orientation(lb1gt_5,112,1348). coords(lb1gt_5,291,111,1348).  visible(lb1gt_5,1348). happensAt(walking(lb1gt_5),1348).
#show holdsAt/2.