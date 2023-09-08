
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

	orientation(mwt2gt_1,153,661). coords(mwt2gt_1,23,134,661).  visible(mwt2gt_1,661). happensAt(walking(mwt2gt_1),661). orientation(mwt2gt_2,152,661). coords(mwt2gt_2,22,138,661).  visible(mwt2gt_2,661). happensAt(walking(mwt2gt_2),661).
orientation(mwt2gt_1,153,662). coords(mwt2gt_1,23,134,662).  visible(mwt2gt_1,662). happensAt(walking(mwt2gt_1),662). orientation(mwt2gt_2,152,662). coords(mwt2gt_2,22,138,662).  visible(mwt2gt_2,662). happensAt(walking(mwt2gt_2),662).
orientation(mwt2gt_1,153,663). coords(mwt2gt_1,23,134,663).  visible(mwt2gt_1,663). happensAt(walking(mwt2gt_1),663). orientation(mwt2gt_2,152,663). coords(mwt2gt_2,24,138,663).  visible(mwt2gt_2,663). happensAt(walking(mwt2gt_2),663).
orientation(mwt2gt_1,153,664). coords(mwt2gt_1,23,134,664).  visible(mwt2gt_1,664). happensAt(walking(mwt2gt_1),664). orientation(mwt2gt_2,152,664). coords(mwt2gt_2,24,138,664).  visible(mwt2gt_2,664). happensAt(walking(mwt2gt_2),664).
orientation(mwt2gt_1,153,665). coords(mwt2gt_1,23,133,665).  visible(mwt2gt_1,665). happensAt(walking(mwt2gt_1),665). orientation(mwt2gt_2,152,665). coords(mwt2gt_2,24,138,665).  visible(mwt2gt_2,665). happensAt(walking(mwt2gt_2),665).
orientation(mwt2gt_1,153,666). coords(mwt2gt_1,23,132,666).  visible(mwt2gt_1,666). happensAt(walking(mwt2gt_1),666). orientation(mwt2gt_2,152,666). coords(mwt2gt_2,24,138,666).  visible(mwt2gt_2,666). happensAt(walking(mwt2gt_2),666).
orientation(mwt2gt_1,153,667). coords(mwt2gt_1,23,132,667).  visible(mwt2gt_1,667). happensAt(walking(mwt2gt_1),667). orientation(mwt2gt_2,152,667). coords(mwt2gt_2,24,138,667).  visible(mwt2gt_2,667). happensAt(walking(mwt2gt_2),667).
orientation(mwt2gt_1,153,668). coords(mwt2gt_1,23,132,668).  visible(mwt2gt_1,668). happensAt(walking(mwt2gt_1),668). orientation(mwt2gt_2,152,668). coords(mwt2gt_2,24,138,668).  visible(mwt2gt_2,668). happensAt(walking(mwt2gt_2),668).
orientation(mwt2gt_1,153,669). coords(mwt2gt_1,23,132,669).  visible(mwt2gt_1,669). happensAt(walking(mwt2gt_1),669). orientation(mwt2gt_2,152,669). coords(mwt2gt_2,24,138,669).  visible(mwt2gt_2,669). happensAt(walking(mwt2gt_2),669).
orientation(mwt2gt_1,153,670). coords(mwt2gt_1,26,132,670).  visible(mwt2gt_1,670). happensAt(walking(mwt2gt_1),670). orientation(mwt2gt_2,152,670). coords(mwt2gt_2,24,138,670).  visible(mwt2gt_2,670). happensAt(walking(mwt2gt_2),670).
orientation(mwt2gt_1,153,671). coords(mwt2gt_1,26,131,671).  visible(mwt2gt_1,671). happensAt(walking(mwt2gt_1),671). orientation(mwt2gt_2,152,671). coords(mwt2gt_2,24,138,671).  visible(mwt2gt_2,671). happensAt(walking(mwt2gt_2),671).
orientation(mwt2gt_1,153,672). coords(mwt2gt_1,26,131,672).  visible(mwt2gt_1,672). happensAt(inactive(mwt2gt_1),672). orientation(mwt2gt_2,152,672). coords(mwt2gt_2,24,138,672).  visible(mwt2gt_2,672). happensAt(inactive(mwt2gt_2),672).
orientation(mwt2gt_1,153,673). coords(mwt2gt_1,26,131,673).  visible(mwt2gt_1,673). happensAt(inactive(mwt2gt_1),673). orientation(mwt2gt_2,152,673). coords(mwt2gt_2,24,138,673).  visible(mwt2gt_2,673). happensAt(inactive(mwt2gt_2),673).
orientation(mwt2gt_1,153,674). coords(mwt2gt_1,26,131,674).  visible(mwt2gt_1,674). happensAt(inactive(mwt2gt_1),674). orientation(mwt2gt_2,152,674). coords(mwt2gt_2,24,138,674).  visible(mwt2gt_2,674). happensAt(inactive(mwt2gt_2),674).
orientation(mwt2gt_1,153,675). coords(mwt2gt_1,26,131,675).  visible(mwt2gt_1,675). happensAt(inactive(mwt2gt_1),675). orientation(mwt2gt_2,152,675). coords(mwt2gt_2,24,138,675).  visible(mwt2gt_2,675). happensAt(inactive(mwt2gt_2),675).
orientation(mwt2gt_1,153,676). coords(mwt2gt_1,26,131,676).  visible(mwt2gt_1,676). happensAt(inactive(mwt2gt_1),676). orientation(mwt2gt_2,152,676). coords(mwt2gt_2,24,138,676).  visible(mwt2gt_2,676). happensAt(inactive(mwt2gt_2),676).
orientation(mwt2gt_1,153,677). coords(mwt2gt_1,26,131,677).  visible(mwt2gt_1,677). happensAt(inactive(mwt2gt_1),677). orientation(mwt2gt_2,152,677). coords(mwt2gt_2,24,138,677).  visible(mwt2gt_2,677). happensAt(inactive(mwt2gt_2),677).
orientation(mwt2gt_1,153,678). coords(mwt2gt_1,26,131,678).  visible(mwt2gt_1,678). happensAt(inactive(mwt2gt_1),678). orientation(mwt2gt_2,152,678). coords(mwt2gt_2,24,138,678).  visible(mwt2gt_2,678). happensAt(inactive(mwt2gt_2),678).
orientation(mwt2gt_1,153,679). coords(mwt2gt_1,26,131,679).  visible(mwt2gt_1,679). happensAt(inactive(mwt2gt_1),679). orientation(mwt2gt_2,152,679). coords(mwt2gt_2,24,138,679).  visible(mwt2gt_2,679). happensAt(inactive(mwt2gt_2),679).
orientation(mwt2gt_1,153,680). coords(mwt2gt_1,26,131,680).  visible(mwt2gt_1,680). happensAt(inactive(mwt2gt_1),680). orientation(mwt2gt_2,152,680). coords(mwt2gt_2,24,138,680).  visible(mwt2gt_2,680). happensAt(inactive(mwt2gt_2),680).
orientation(mwt2gt_1,153,681). coords(mwt2gt_1,26,130,681).  visible(mwt2gt_1,681). happensAt(inactive(mwt2gt_1),681). orientation(mwt2gt_2,152,681). coords(mwt2gt_2,24,138,681).  visible(mwt2gt_2,681). happensAt(inactive(mwt2gt_2),681).
orientation(mwt2gt_1,153,682). coords(mwt2gt_1,25,130,682).  visible(mwt2gt_1,682). happensAt(inactive(mwt2gt_1),682). orientation(mwt2gt_2,152,682). coords(mwt2gt_2,24,138,682).  visible(mwt2gt_2,682). happensAt(inactive(mwt2gt_2),682).
orientation(mwt2gt_1,153,683). coords(mwt2gt_1,26,131,683).  visible(mwt2gt_1,683). happensAt(inactive(mwt2gt_1),683). orientation(mwt2gt_2,152,683). coords(mwt2gt_2,24,138,683).  visible(mwt2gt_2,683). happensAt(inactive(mwt2gt_2),683).
orientation(mwt2gt_1,153,684). coords(mwt2gt_1,26,130,684).  visible(mwt2gt_1,684). happensAt(inactive(mwt2gt_1),684). orientation(mwt2gt_2,152,684). coords(mwt2gt_2,24,138,684).  visible(mwt2gt_2,684). happensAt(inactive(mwt2gt_2),684).
orientation(mwt2gt_1,153,685). coords(mwt2gt_1,26,131,685).  visible(mwt2gt_1,685). happensAt(inactive(mwt2gt_1),685). orientation(mwt2gt_2,152,685). coords(mwt2gt_2,24,138,685).  visible(mwt2gt_2,685). happensAt(inactive(mwt2gt_2),685).
#show holdsAt/2.