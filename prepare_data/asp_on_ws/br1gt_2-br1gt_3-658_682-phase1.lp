
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

	orientation(br1gt_2,12,658). coords(br1gt_2,86,209,658).  visible(br1gt_2,658). happensAt(inactive(br1gt_2),658). orientation(br1gt_3,119,658). coords(br1gt_3,298,91,658).  visible(br1gt_3,658). happensAt(active(br1gt_3),658).
orientation(br1gt_2,12,659). coords(br1gt_2,86,209,659).  visible(br1gt_2,659). happensAt(inactive(br1gt_2),659). orientation(br1gt_3,119,659). coords(br1gt_3,298,91,659).  visible(br1gt_3,659). happensAt(active(br1gt_3),659).
orientation(br1gt_2,12,660). coords(br1gt_2,86,209,660).  visible(br1gt_2,660). happensAt(inactive(br1gt_2),660). orientation(br1gt_3,119,660). coords(br1gt_3,298,91,660).  visible(br1gt_3,660). happensAt(active(br1gt_3),660).
orientation(br1gt_2,12,661). coords(br1gt_2,86,209,661).  visible(br1gt_2,661). happensAt(inactive(br1gt_2),661). orientation(br1gt_3,119,661). coords(br1gt_3,298,91,661).  visible(br1gt_3,661). happensAt(active(br1gt_3),661).
orientation(br1gt_2,12,662). coords(br1gt_2,86,209,662).  visible(br1gt_2,662). happensAt(inactive(br1gt_2),662). orientation(br1gt_3,119,662). coords(br1gt_3,298,91,662).  visible(br1gt_3,662). happensAt(active(br1gt_3),662).
orientation(br1gt_2,12,663). coords(br1gt_2,86,209,663).  visible(br1gt_2,663). happensAt(inactive(br1gt_2),663). orientation(br1gt_3,119,663). coords(br1gt_3,298,92,663).  visible(br1gt_3,663). happensAt(active(br1gt_3),663).
orientation(br1gt_2,12,664). coords(br1gt_2,86,209,664).  visible(br1gt_2,664). happensAt(inactive(br1gt_2),664). orientation(br1gt_3,119,664). coords(br1gt_3,298,92,664).  visible(br1gt_3,664). happensAt(active(br1gt_3),664).
orientation(br1gt_2,12,665). coords(br1gt_2,86,209,665).  visible(br1gt_2,665). happensAt(inactive(br1gt_2),665). orientation(br1gt_3,114,665). coords(br1gt_3,300,92,665).  visible(br1gt_3,665). happensAt(active(br1gt_3),665).
orientation(br1gt_2,12,666). coords(br1gt_2,86,209,666).  visible(br1gt_2,666). happensAt(inactive(br1gt_2),666). orientation(br1gt_3,114,666). coords(br1gt_3,301,92,666).  visible(br1gt_3,666). happensAt(active(br1gt_3),666).
orientation(br1gt_2,12,667). coords(br1gt_2,86,209,667).  visible(br1gt_2,667). happensAt(inactive(br1gt_2),667). orientation(br1gt_3,114,667). coords(br1gt_3,301,92,667).  visible(br1gt_3,667). happensAt(active(br1gt_3),667).
orientation(br1gt_2,12,668). coords(br1gt_2,86,209,668).  visible(br1gt_2,668). happensAt(inactive(br1gt_2),668). orientation(br1gt_3,114,668). coords(br1gt_3,301,92,668).  visible(br1gt_3,668). happensAt(active(br1gt_3),668).
orientation(br1gt_2,12,669). coords(br1gt_2,86,209,669).  visible(br1gt_2,669). happensAt(inactive(br1gt_2),669). orientation(br1gt_3,114,669). coords(br1gt_3,301,92,669).  visible(br1gt_3,669). happensAt(active(br1gt_3),669).
orientation(br1gt_2,12,670). coords(br1gt_2,86,209,670).  visible(br1gt_2,670). happensAt(inactive(br1gt_2),670). orientation(br1gt_3,114,670). coords(br1gt_3,302,93,670).  visible(br1gt_3,670). happensAt(active(br1gt_3),670).
orientation(br1gt_2,12,671). coords(br1gt_2,86,209,671).  visible(br1gt_2,671). happensAt(inactive(br1gt_2),671). orientation(br1gt_3,105,671). coords(br1gt_3,302,96,671).  visible(br1gt_3,671). happensAt(active(br1gt_3),671).
orientation(br1gt_2,12,672). coords(br1gt_2,86,209,672).  visible(br1gt_2,672). happensAt(inactive(br1gt_2),672). orientation(br1gt_3,105,672). coords(br1gt_3,300,97,672).  visible(br1gt_3,672). happensAt(active(br1gt_3),672).
orientation(br1gt_2,12,673). coords(br1gt_2,86,209,673).  visible(br1gt_2,673). happensAt(inactive(br1gt_2),673). orientation(br1gt_3,105,673). coords(br1gt_3,300,97,673).  visible(br1gt_3,673). happensAt(active(br1gt_3),673).
orientation(br1gt_2,12,674). coords(br1gt_2,86,209,674).  visible(br1gt_2,674). happensAt(inactive(br1gt_2),674). orientation(br1gt_3,105,674). coords(br1gt_3,299,98,674).  visible(br1gt_3,674). happensAt(active(br1gt_3),674).
orientation(br1gt_2,12,675). coords(br1gt_2,86,209,675).  visible(br1gt_2,675). happensAt(inactive(br1gt_2),675). orientation(br1gt_3,102,675). coords(br1gt_3,299,98,675).  visible(br1gt_3,675). happensAt(active(br1gt_3),675).
orientation(br1gt_2,12,676). coords(br1gt_2,86,209,676).  visible(br1gt_2,676). happensAt(inactive(br1gt_2),676). orientation(br1gt_3,102,676). coords(br1gt_3,299,98,676).  visible(br1gt_3,676). happensAt(active(br1gt_3),676).
orientation(br1gt_2,12,677). coords(br1gt_2,86,209,677).  visible(br1gt_2,677). happensAt(inactive(br1gt_2),677). orientation(br1gt_3,102,677). coords(br1gt_3,296,98,677).  visible(br1gt_3,677). happensAt(active(br1gt_3),677).
orientation(br1gt_2,12,678). coords(br1gt_2,86,209,678).  visible(br1gt_2,678). happensAt(inactive(br1gt_2),678). orientation(br1gt_3,102,678). coords(br1gt_3,296,98,678).  visible(br1gt_3,678). happensAt(active(br1gt_3),678).
orientation(br1gt_2,12,679). coords(br1gt_2,86,209,679).  visible(br1gt_2,679). happensAt(inactive(br1gt_2),679). orientation(br1gt_3,102,679). coords(br1gt_3,296,98,679).  visible(br1gt_3,679). happensAt(active(br1gt_3),679).
orientation(br1gt_2,12,680). coords(br1gt_2,86,209,680).  visible(br1gt_2,680). happensAt(inactive(br1gt_2),680). orientation(br1gt_3,102,680). coords(br1gt_3,295,97,680).  visible(br1gt_3,680). happensAt(active(br1gt_3),680).
orientation(br1gt_2,12,681). coords(br1gt_2,86,209,681).  visible(br1gt_2,681). happensAt(inactive(br1gt_2),681). orientation(br1gt_3,102,681). coords(br1gt_3,295,97,681).  visible(br1gt_3,681). happensAt(active(br1gt_3),681).
orientation(br1gt_2,12,682). coords(br1gt_2,86,209,682).  visible(br1gt_2,682). happensAt(inactive(br1gt_2),682). orientation(br1gt_3,96,682). coords(br1gt_3,295,97,682).  visible(br1gt_3,682). happensAt(active(br1gt_3),682).
#show holdsAt/2.