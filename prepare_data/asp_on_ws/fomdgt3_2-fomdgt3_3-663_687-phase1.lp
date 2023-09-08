
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

	orientation(fomdgt3_2,110,663). coords(fomdgt3_2,288,218,663).  visible(fomdgt3_2,663). happensAt(walking(fomdgt3_2),663). orientation(fomdgt3_3,159,663). coords(fomdgt3_3,95,97,663).  visible(fomdgt3_3,663). happensAt(walking(fomdgt3_3),663).
orientation(fomdgt3_2,110,664). coords(fomdgt3_2,290,220,664).  visible(fomdgt3_2,664). happensAt(walking(fomdgt3_2),664). orientation(fomdgt3_3,159,664). coords(fomdgt3_3,95,97,664).  visible(fomdgt3_3,664). happensAt(walking(fomdgt3_3),664).
orientation(fomdgt3_2,110,665). coords(fomdgt3_2,291,222,665).  visible(fomdgt3_2,665). happensAt(walking(fomdgt3_2),665). orientation(fomdgt3_3,159,665). coords(fomdgt3_3,96,98,665).  visible(fomdgt3_3,665). happensAt(walking(fomdgt3_3),665).
orientation(fomdgt3_2,110,666). coords(fomdgt3_2,292,223,666).  visible(fomdgt3_2,666). happensAt(walking(fomdgt3_2),666). orientation(fomdgt3_3,159,666). coords(fomdgt3_3,96,98,666).  visible(fomdgt3_3,666). happensAt(walking(fomdgt3_3),666).
orientation(fomdgt3_2,110,667). coords(fomdgt3_2,293,226,667).  visible(fomdgt3_2,667). happensAt(walking(fomdgt3_2),667). orientation(fomdgt3_3,159,667). coords(fomdgt3_3,98,98,667).  visible(fomdgt3_3,667). happensAt(walking(fomdgt3_3),667).
orientation(fomdgt3_2,110,668). coords(fomdgt3_2,293,228,668).  visible(fomdgt3_2,668). happensAt(walking(fomdgt3_2),668). orientation(fomdgt3_3,159,668). coords(fomdgt3_3,99,98,668).  visible(fomdgt3_3,668). happensAt(walking(fomdgt3_3),668).
orientation(fomdgt3_2,110,669). coords(fomdgt3_2,293,228,669).  visible(fomdgt3_2,669). happensAt(walking(fomdgt3_2),669). orientation(fomdgt3_3,159,669). coords(fomdgt3_3,99,98,669).  visible(fomdgt3_3,669). happensAt(walking(fomdgt3_3),669).
orientation(fomdgt3_2,110,670). coords(fomdgt3_2,293,230,670).  visible(fomdgt3_2,670). happensAt(walking(fomdgt3_2),670). orientation(fomdgt3_3,159,670). coords(fomdgt3_3,99,99,670).  visible(fomdgt3_3,670). happensAt(walking(fomdgt3_3),670).
orientation(fomdgt3_2,110,671). coords(fomdgt3_2,293,230,671).  visible(fomdgt3_2,671). happensAt(walking(fomdgt3_2),671). orientation(fomdgt3_3,159,671). coords(fomdgt3_3,99,99,671).  visible(fomdgt3_3,671). happensAt(walking(fomdgt3_3),671).
orientation(fomdgt3_2,110,672). coords(fomdgt3_2,293,230,672).  visible(fomdgt3_2,672). happensAt(walking(fomdgt3_2),672). orientation(fomdgt3_3,159,672). coords(fomdgt3_3,99,99,672).  visible(fomdgt3_3,672). happensAt(walking(fomdgt3_3),672).
orientation(fomdgt3_2,110,673). coords(fomdgt3_2,293,231,673).  visible(fomdgt3_2,673). happensAt(walking(fomdgt3_2),673). orientation(fomdgt3_3,159,673). coords(fomdgt3_3,100,101,673).  visible(fomdgt3_3,673). happensAt(walking(fomdgt3_3),673).
orientation(fomdgt3_2,110,674). coords(fomdgt3_2,293,235,674).  visible(fomdgt3_2,674). happensAt(walking(fomdgt3_2),674). orientation(fomdgt3_3,159,674). coords(fomdgt3_3,100,101,674).  visible(fomdgt3_3,674). happensAt(walking(fomdgt3_3),674).
orientation(fomdgt3_2,110,675). coords(fomdgt3_2,294,237,675).  visible(fomdgt3_2,675). happensAt(walking(fomdgt3_2),675). orientation(fomdgt3_3,159,675). coords(fomdgt3_3,103,103,675).  visible(fomdgt3_3,675). happensAt(walking(fomdgt3_3),675).
orientation(fomdgt3_2,110,676). coords(fomdgt3_2,294,241,676).  visible(fomdgt3_2,676). happensAt(walking(fomdgt3_2),676). orientation(fomdgt3_3,159,676). coords(fomdgt3_3,105,103,676).  visible(fomdgt3_3,676). happensAt(walking(fomdgt3_3),676).
orientation(fomdgt3_2,110,677). coords(fomdgt3_2,294,244,677).  visible(fomdgt3_2,677). happensAt(walking(fomdgt3_2),677). orientation(fomdgt3_3,159,677). coords(fomdgt3_3,105,103,677).  visible(fomdgt3_3,677). happensAt(walking(fomdgt3_3),677).
orientation(fomdgt3_2,110,678). coords(fomdgt3_2,294,246,678).  visible(fomdgt3_2,678). happensAt(walking(fomdgt3_2),678). orientation(fomdgt3_3,159,678). coords(fomdgt3_3,106,105,678).  visible(fomdgt3_3,678). happensAt(walking(fomdgt3_3),678).
orientation(fomdgt3_2,110,679). coords(fomdgt3_2,294,246,679).  visible(fomdgt3_2,679). happensAt(walking(fomdgt3_2),679). orientation(fomdgt3_3,159,679). coords(fomdgt3_3,106,105,679).  visible(fomdgt3_3,679). happensAt(walking(fomdgt3_3),679).
orientation(fomdgt3_2,110,680). coords(fomdgt3_2,294,250,680).  visible(fomdgt3_2,680). happensAt(walking(fomdgt3_2),680). orientation(fomdgt3_3,159,680). coords(fomdgt3_3,106,105,680).  visible(fomdgt3_3,680). happensAt(walking(fomdgt3_3),680).
orientation(fomdgt3_2,110,681). coords(fomdgt3_2,294,252,681).  visible(fomdgt3_2,681). happensAt(walking(fomdgt3_2),681). orientation(fomdgt3_3,159,681). coords(fomdgt3_3,106,105,681).  visible(fomdgt3_3,681). happensAt(walking(fomdgt3_3),681).
orientation(fomdgt3_2,110,682). coords(fomdgt3_2,294,253,682).  visible(fomdgt3_2,682). happensAt(walking(fomdgt3_2),682). orientation(fomdgt3_3,159,682). coords(fomdgt3_3,106,105,682).  visible(fomdgt3_3,682). happensAt(walking(fomdgt3_3),682).
orientation(fomdgt3_2,110,683). coords(fomdgt3_2,294,255,683).  visible(fomdgt3_2,683). happensAt(walking(fomdgt3_2),683). orientation(fomdgt3_3,159,683). coords(fomdgt3_3,107,105,683).  visible(fomdgt3_3,683). happensAt(walking(fomdgt3_3),683).
orientation(fomdgt3_2,110,684). coords(fomdgt3_2,294,256,684).  visible(fomdgt3_2,684). happensAt(walking(fomdgt3_2),684). orientation(fomdgt3_3,159,684). coords(fomdgt3_3,109,105,684).  visible(fomdgt3_3,684). happensAt(walking(fomdgt3_3),684).
orientation(fomdgt3_2,110,685). coords(fomdgt3_2,294,257,685).  visible(fomdgt3_2,685). happensAt(walking(fomdgt3_2),685). orientation(fomdgt3_3,159,685). coords(fomdgt3_3,109,105,685).  visible(fomdgt3_3,685). happensAt(walking(fomdgt3_3),685).
orientation(fomdgt3_2,80,686). coords(fomdgt3_2,294,258,686).  visible(fomdgt3_2,686). happensAt(walking(fomdgt3_2),686). orientation(fomdgt3_3,159,686). coords(fomdgt3_3,113,106,686).  visible(fomdgt3_3,686). happensAt(walking(fomdgt3_3),686).
orientation(fomdgt3_2,80,687). coords(fomdgt3_2,294,259,687).  visible(fomdgt3_2,687). happensAt(walking(fomdgt3_2),687). orientation(fomdgt3_3,159,687). coords(fomdgt3_3,114,107,687).  visible(fomdgt3_3,687). happensAt(walking(fomdgt3_3),687).
#show holdsAt/2.