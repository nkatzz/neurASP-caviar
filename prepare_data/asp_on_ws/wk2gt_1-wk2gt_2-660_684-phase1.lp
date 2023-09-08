
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

	orientation(wk2gt_1,125,660). coords(wk2gt_1,114,53,660).  visible(wk2gt_1,660). happensAt(inactive(wk2gt_1),660). orientation(wk2gt_2,142,660). coords(wk2gt_2,99,63,660).  visible(wk2gt_2,660). happensAt(inactive(wk2gt_2),660).
orientation(wk2gt_1,125,661). coords(wk2gt_1,114,53,661).  visible(wk2gt_1,661). happensAt(inactive(wk2gt_1),661). orientation(wk2gt_2,142,661). coords(wk2gt_2,99,63,661).  visible(wk2gt_2,661). happensAt(inactive(wk2gt_2),661).
orientation(wk2gt_1,125,662). coords(wk2gt_1,114,53,662).  visible(wk2gt_1,662). happensAt(inactive(wk2gt_1),662). orientation(wk2gt_2,142,662). coords(wk2gt_2,99,63,662).  visible(wk2gt_2,662). happensAt(inactive(wk2gt_2),662).
orientation(wk2gt_1,125,663). coords(wk2gt_1,114,53,663).  visible(wk2gt_1,663). happensAt(inactive(wk2gt_1),663). orientation(wk2gt_2,142,663). coords(wk2gt_2,99,63,663).  visible(wk2gt_2,663). happensAt(inactive(wk2gt_2),663).
orientation(wk2gt_1,125,664). coords(wk2gt_1,114,53,664).  visible(wk2gt_1,664). happensAt(inactive(wk2gt_1),664). orientation(wk2gt_2,142,664). coords(wk2gt_2,99,63,664).  visible(wk2gt_2,664). happensAt(inactive(wk2gt_2),664).
orientation(wk2gt_1,125,665). coords(wk2gt_1,114,53,665).  visible(wk2gt_1,665). happensAt(inactive(wk2gt_1),665). orientation(wk2gt_2,142,665). coords(wk2gt_2,99,63,665).  visible(wk2gt_2,665). happensAt(inactive(wk2gt_2),665).
orientation(wk2gt_1,125,666). coords(wk2gt_1,114,53,666).  visible(wk2gt_1,666). happensAt(inactive(wk2gt_1),666). orientation(wk2gt_2,142,666). coords(wk2gt_2,99,63,666).  visible(wk2gt_2,666). happensAt(inactive(wk2gt_2),666).
orientation(wk2gt_1,125,667). coords(wk2gt_1,114,53,667).  visible(wk2gt_1,667). happensAt(inactive(wk2gt_1),667). orientation(wk2gt_2,142,667). coords(wk2gt_2,99,63,667).  visible(wk2gt_2,667). happensAt(inactive(wk2gt_2),667).
orientation(wk2gt_1,125,668). coords(wk2gt_1,114,53,668).  visible(wk2gt_1,668). happensAt(inactive(wk2gt_1),668). orientation(wk2gt_2,142,668). coords(wk2gt_2,99,63,668).  visible(wk2gt_2,668). happensAt(inactive(wk2gt_2),668).
orientation(wk2gt_1,125,669). coords(wk2gt_1,114,53,669).  visible(wk2gt_1,669). happensAt(inactive(wk2gt_1),669). orientation(wk2gt_2,142,669). coords(wk2gt_2,99,63,669).  visible(wk2gt_2,669). happensAt(inactive(wk2gt_2),669).
orientation(wk2gt_1,125,670). coords(wk2gt_1,114,53,670).  visible(wk2gt_1,670). happensAt(inactive(wk2gt_1),670). orientation(wk2gt_2,142,670). coords(wk2gt_2,99,63,670).  visible(wk2gt_2,670). happensAt(inactive(wk2gt_2),670).
orientation(wk2gt_1,125,671). coords(wk2gt_1,114,53,671).  visible(wk2gt_1,671). happensAt(inactive(wk2gt_1),671). orientation(wk2gt_2,142,671). coords(wk2gt_2,99,63,671).  visible(wk2gt_2,671). happensAt(inactive(wk2gt_2),671).
orientation(wk2gt_1,125,672). coords(wk2gt_1,114,53,672).  visible(wk2gt_1,672). happensAt(inactive(wk2gt_1),672). orientation(wk2gt_2,142,672). coords(wk2gt_2,99,63,672).  visible(wk2gt_2,672). happensAt(inactive(wk2gt_2),672).
orientation(wk2gt_1,125,673). coords(wk2gt_1,114,53,673).  visible(wk2gt_1,673). happensAt(inactive(wk2gt_1),673). orientation(wk2gt_2,142,673). coords(wk2gt_2,99,63,673).  visible(wk2gt_2,673). happensAt(inactive(wk2gt_2),673).
orientation(wk2gt_1,125,674). coords(wk2gt_1,114,53,674).  visible(wk2gt_1,674). happensAt(inactive(wk2gt_1),674). orientation(wk2gt_2,142,674). coords(wk2gt_2,99,63,674).  visible(wk2gt_2,674). happensAt(inactive(wk2gt_2),674).
orientation(wk2gt_1,125,675). coords(wk2gt_1,114,53,675).  visible(wk2gt_1,675). happensAt(inactive(wk2gt_1),675). orientation(wk2gt_2,142,675). coords(wk2gt_2,99,63,675).  visible(wk2gt_2,675). happensAt(inactive(wk2gt_2),675).
orientation(wk2gt_1,125,676). coords(wk2gt_1,114,53,676).  visible(wk2gt_1,676). happensAt(inactive(wk2gt_1),676). orientation(wk2gt_2,142,676). coords(wk2gt_2,99,63,676).  visible(wk2gt_2,676). happensAt(inactive(wk2gt_2),676).
orientation(wk2gt_1,125,677). coords(wk2gt_1,114,53,677).  visible(wk2gt_1,677). happensAt(inactive(wk2gt_1),677). orientation(wk2gt_2,142,677). coords(wk2gt_2,99,63,677).  visible(wk2gt_2,677). happensAt(inactive(wk2gt_2),677).
orientation(wk2gt_1,125,678). coords(wk2gt_1,114,53,678).  visible(wk2gt_1,678). happensAt(inactive(wk2gt_1),678). orientation(wk2gt_2,142,678). coords(wk2gt_2,99,63,678).  visible(wk2gt_2,678). happensAt(inactive(wk2gt_2),678).
orientation(wk2gt_1,125,679). coords(wk2gt_1,114,53,679).  visible(wk2gt_1,679). happensAt(inactive(wk2gt_1),679). orientation(wk2gt_2,142,679). coords(wk2gt_2,99,63,679).  visible(wk2gt_2,679). happensAt(inactive(wk2gt_2),679).
orientation(wk2gt_1,125,680). coords(wk2gt_1,114,53,680).  visible(wk2gt_1,680). happensAt(inactive(wk2gt_1),680). orientation(wk2gt_2,142,680). coords(wk2gt_2,99,63,680).  visible(wk2gt_2,680). happensAt(inactive(wk2gt_2),680).
orientation(wk2gt_1,125,681). coords(wk2gt_1,114,53,681).  visible(wk2gt_1,681). happensAt(inactive(wk2gt_1),681). orientation(wk2gt_2,142,681). coords(wk2gt_2,99,63,681).  visible(wk2gt_2,681). happensAt(inactive(wk2gt_2),681).
orientation(wk2gt_1,125,682). coords(wk2gt_1,114,53,682).  visible(wk2gt_1,682). happensAt(inactive(wk2gt_1),682). orientation(wk2gt_2,142,682). coords(wk2gt_2,99,63,682).  visible(wk2gt_2,682). happensAt(inactive(wk2gt_2),682).
orientation(wk2gt_1,125,683). coords(wk2gt_1,114,53,683).  visible(wk2gt_1,683). happensAt(inactive(wk2gt_1),683). orientation(wk2gt_2,142,683). coords(wk2gt_2,99,63,683).  visible(wk2gt_2,683). happensAt(inactive(wk2gt_2),683).
orientation(wk2gt_1,125,684). coords(wk2gt_1,114,53,684).  visible(wk2gt_1,684). happensAt(inactive(wk2gt_1),684). orientation(wk2gt_2,142,684). coords(wk2gt_2,99,63,684).  visible(wk2gt_2,684). happensAt(inactive(wk2gt_2),684).
#show holdsAt/2.