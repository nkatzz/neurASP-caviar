
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

	orientation(mwt1gt_5,150,661). coords(mwt1gt_5,84,80,661).  visible(mwt1gt_5,661). happensAt(walking(mwt1gt_5),661). orientation(mwt1gt_6,150,661). coords(mwt1gt_6,75,84,661).  visible(mwt1gt_6,661). happensAt(walking(mwt1gt_6),661).
orientation(mwt1gt_5,150,662). coords(mwt1gt_5,84,80,662).  visible(mwt1gt_5,662). happensAt(walking(mwt1gt_5),662). orientation(mwt1gt_6,150,662). coords(mwt1gt_6,75,84,662).  visible(mwt1gt_6,662). happensAt(walking(mwt1gt_6),662).
orientation(mwt1gt_5,150,663). coords(mwt1gt_5,84,80,663).  visible(mwt1gt_5,663). happensAt(walking(mwt1gt_5),663). orientation(mwt1gt_6,150,663). coords(mwt1gt_6,75,84,663).  visible(mwt1gt_6,663). happensAt(walking(mwt1gt_6),663).
orientation(mwt1gt_5,150,664). coords(mwt1gt_5,84,80,664).  visible(mwt1gt_5,664). happensAt(walking(mwt1gt_5),664). orientation(mwt1gt_6,150,664). coords(mwt1gt_6,75,84,664).  visible(mwt1gt_6,664). happensAt(walking(mwt1gt_6),664).
orientation(mwt1gt_5,150,665). coords(mwt1gt_5,84,80,665).  visible(mwt1gt_5,665). happensAt(walking(mwt1gt_5),665). orientation(mwt1gt_6,150,665). coords(mwt1gt_6,75,84,665).  visible(mwt1gt_6,665). happensAt(walking(mwt1gt_6),665).
orientation(mwt1gt_5,150,666). coords(mwt1gt_5,84,80,666).  visible(mwt1gt_5,666). happensAt(walking(mwt1gt_5),666). orientation(mwt1gt_6,150,666). coords(mwt1gt_6,75,84,666).  visible(mwt1gt_6,666). happensAt(walking(mwt1gt_6),666).
orientation(mwt1gt_5,150,667). coords(mwt1gt_5,84,82,667).  visible(mwt1gt_5,667). happensAt(walking(mwt1gt_5),667). orientation(mwt1gt_6,150,667). coords(mwt1gt_6,75,86,667).  visible(mwt1gt_6,667). happensAt(walking(mwt1gt_6),667).
orientation(mwt1gt_5,150,668). coords(mwt1gt_5,84,82,668).  visible(mwt1gt_5,668). happensAt(walking(mwt1gt_5),668). orientation(mwt1gt_6,150,668). coords(mwt1gt_6,75,86,668).  visible(mwt1gt_6,668). happensAt(walking(mwt1gt_6),668).
orientation(mwt1gt_5,150,669). coords(mwt1gt_5,85,83,669).  visible(mwt1gt_5,669). happensAt(walking(mwt1gt_5),669). orientation(mwt1gt_6,150,669). coords(mwt1gt_6,75,89,669).  visible(mwt1gt_6,669). happensAt(walking(mwt1gt_6),669).
orientation(mwt1gt_5,150,670). coords(mwt1gt_5,85,83,670).  visible(mwt1gt_5,670). happensAt(walking(mwt1gt_5),670). orientation(mwt1gt_6,150,670). coords(mwt1gt_6,75,89,670).  visible(mwt1gt_6,670). happensAt(walking(mwt1gt_6),670).
orientation(mwt1gt_5,150,671). coords(mwt1gt_5,85,83,671).  visible(mwt1gt_5,671). happensAt(walking(mwt1gt_5),671). orientation(mwt1gt_6,150,671). coords(mwt1gt_6,76,90,671).  visible(mwt1gt_6,671). happensAt(walking(mwt1gt_6),671).
orientation(mwt1gt_5,150,672). coords(mwt1gt_5,85,83,672).  visible(mwt1gt_5,672). happensAt(walking(mwt1gt_5),672). orientation(mwt1gt_6,150,672). coords(mwt1gt_6,76,90,672).  visible(mwt1gt_6,672). happensAt(walking(mwt1gt_6),672).
orientation(mwt1gt_5,150,673). coords(mwt1gt_5,85,83,673).  visible(mwt1gt_5,673). happensAt(walking(mwt1gt_5),673). orientation(mwt1gt_6,150,673). coords(mwt1gt_6,76,90,673).  visible(mwt1gt_6,673). happensAt(walking(mwt1gt_6),673).
orientation(mwt1gt_5,150,674). coords(mwt1gt_5,85,83,674).  visible(mwt1gt_5,674). happensAt(walking(mwt1gt_5),674). orientation(mwt1gt_6,150,674). coords(mwt1gt_6,76,92,674).  visible(mwt1gt_6,674). happensAt(walking(mwt1gt_6),674).
orientation(mwt1gt_5,150,675). coords(mwt1gt_5,85,83,675).  visible(mwt1gt_5,675). happensAt(walking(mwt1gt_5),675). orientation(mwt1gt_6,150,675). coords(mwt1gt_6,76,92,675).  visible(mwt1gt_6,675). happensAt(walking(mwt1gt_6),675).
orientation(mwt1gt_5,150,676). coords(mwt1gt_5,85,83,676).  visible(mwt1gt_5,676). happensAt(walking(mwt1gt_5),676). orientation(mwt1gt_6,150,676). coords(mwt1gt_6,76,92,676).  visible(mwt1gt_6,676). happensAt(walking(mwt1gt_6),676).
orientation(mwt1gt_5,150,677). coords(mwt1gt_5,85,83,677).  visible(mwt1gt_5,677). happensAt(walking(mwt1gt_5),677). orientation(mwt1gt_6,150,677). coords(mwt1gt_6,76,92,677).  visible(mwt1gt_6,677). happensAt(walking(mwt1gt_6),677).
orientation(mwt1gt_5,150,678). coords(mwt1gt_5,85,83,678).  visible(mwt1gt_5,678). happensAt(walking(mwt1gt_5),678). orientation(mwt1gt_6,150,678). coords(mwt1gt_6,76,92,678).  visible(mwt1gt_6,678). happensAt(walking(mwt1gt_6),678).
orientation(mwt1gt_5,150,679). coords(mwt1gt_5,87,83,679).  visible(mwt1gt_5,679). happensAt(walking(mwt1gt_5),679). orientation(mwt1gt_6,150,679). coords(mwt1gt_6,76,93,679).  visible(mwt1gt_6,679). happensAt(walking(mwt1gt_6),679).
orientation(mwt1gt_5,150,680). coords(mwt1gt_5,89,83,680).  visible(mwt1gt_5,680). happensAt(walking(mwt1gt_5),680). orientation(mwt1gt_6,150,680). coords(mwt1gt_6,76,93,680).  visible(mwt1gt_6,680). happensAt(walking(mwt1gt_6),680).
orientation(mwt1gt_5,150,681). coords(mwt1gt_5,89,83,681).  visible(mwt1gt_5,681). happensAt(walking(mwt1gt_5),681). orientation(mwt1gt_6,150,681). coords(mwt1gt_6,76,93,681).  visible(mwt1gt_6,681). happensAt(walking(mwt1gt_6),681).
orientation(mwt1gt_5,150,682). coords(mwt1gt_5,89,83,682).  visible(mwt1gt_5,682). happensAt(walking(mwt1gt_5),682). orientation(mwt1gt_6,150,682). coords(mwt1gt_6,76,93,682).  visible(mwt1gt_6,682). happensAt(walking(mwt1gt_6),682).
orientation(mwt1gt_5,150,683). coords(mwt1gt_5,91,83,683).  visible(mwt1gt_5,683). happensAt(walking(mwt1gt_5),683). orientation(mwt1gt_6,150,683). coords(mwt1gt_6,77,94,683).  visible(mwt1gt_6,683). happensAt(walking(mwt1gt_6),683).
orientation(mwt1gt_5,150,684). coords(mwt1gt_5,91,83,684).  visible(mwt1gt_5,684). happensAt(walking(mwt1gt_5),684). orientation(mwt1gt_6,150,684). coords(mwt1gt_6,77,94,684).  visible(mwt1gt_6,684). happensAt(walking(mwt1gt_6),684).
orientation(mwt1gt_5,150,685). coords(mwt1gt_5,92,83,685).  visible(mwt1gt_5,685). happensAt(walking(mwt1gt_5),685). orientation(mwt1gt_6,150,685). coords(mwt1gt_6,78,95,685).  visible(mwt1gt_6,685). happensAt(walking(mwt1gt_6),685).
#show holdsAt/2.