
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

	orientation(br3gt_1,131,654). coords(br3gt_1,184,47,654).  visible(br3gt_1,654). happensAt(active(br3gt_1),654). orientation(br3gt_3,150,654). coords(br3gt_3,4,94,654).  happensAt(appear(br3gt_3),654). happensAt(walking(br3gt_3),654).
orientation(br3gt_1,131,655). coords(br3gt_1,184,47,655).  visible(br3gt_1,655). happensAt(active(br3gt_1),655). orientation(br3gt_3,150,655). coords(br3gt_3,4,94,655).  visible(br3gt_3,655). happensAt(walking(br3gt_3),655).
orientation(br3gt_1,131,656). coords(br3gt_1,184,47,656).  visible(br3gt_1,656). happensAt(active(br3gt_1),656). orientation(br3gt_3,150,656). coords(br3gt_3,4,92,656).  visible(br3gt_3,656). happensAt(walking(br3gt_3),656).
orientation(br3gt_1,131,657). coords(br3gt_1,184,47,657).  visible(br3gt_1,657). happensAt(active(br3gt_1),657). orientation(br3gt_3,150,657). coords(br3gt_3,4,92,657).  visible(br3gt_3,657). happensAt(walking(br3gt_3),657).
orientation(br3gt_1,131,658). coords(br3gt_1,184,47,658).  visible(br3gt_1,658). happensAt(active(br3gt_1),658). orientation(br3gt_3,150,658). coords(br3gt_3,4,91,658).  visible(br3gt_3,658). happensAt(walking(br3gt_3),658).
orientation(br3gt_1,131,659). coords(br3gt_1,184,47,659).  visible(br3gt_1,659). happensAt(active(br3gt_1),659). orientation(br3gt_3,150,659). coords(br3gt_3,4,90,659).  visible(br3gt_3,659). happensAt(walking(br3gt_3),659).
orientation(br3gt_1,131,660). coords(br3gt_1,184,47,660).  visible(br3gt_1,660). happensAt(walking(br3gt_1),660). orientation(br3gt_3,150,660). coords(br3gt_3,4,90,660).  visible(br3gt_3,660). happensAt(walking(br3gt_3),660).
orientation(br3gt_1,131,661). coords(br3gt_1,184,47,661).  visible(br3gt_1,661). happensAt(walking(br3gt_1),661). orientation(br3gt_3,150,661). coords(br3gt_3,4,90,661).  visible(br3gt_3,661). happensAt(walking(br3gt_3),661).
orientation(br3gt_1,131,662). coords(br3gt_1,184,46,662).  visible(br3gt_1,662). happensAt(walking(br3gt_1),662). orientation(br3gt_3,150,662). coords(br3gt_3,4,90,662).  visible(br3gt_3,662). happensAt(walking(br3gt_3),662).
orientation(br3gt_1,131,663). coords(br3gt_1,184,46,663).  visible(br3gt_1,663). happensAt(walking(br3gt_1),663). orientation(br3gt_3,150,663). coords(br3gt_3,4,90,663).  visible(br3gt_3,663). happensAt(walking(br3gt_3),663).
orientation(br3gt_1,131,664). coords(br3gt_1,185,46,664).  visible(br3gt_1,664). happensAt(walking(br3gt_1),664). orientation(br3gt_3,150,664). coords(br3gt_3,4,90,664).  visible(br3gt_3,664). happensAt(walking(br3gt_3),664).
orientation(br3gt_1,131,665). coords(br3gt_1,185,46,665).  visible(br3gt_1,665). happensAt(walking(br3gt_1),665). orientation(br3gt_3,150,665). coords(br3gt_3,4,89,665).  visible(br3gt_3,665). happensAt(walking(br3gt_3),665).
orientation(br3gt_1,131,666). coords(br3gt_1,185,46,666).  visible(br3gt_1,666). happensAt(walking(br3gt_1),666). orientation(br3gt_3,150,666). coords(br3gt_3,4,89,666).  visible(br3gt_3,666). happensAt(walking(br3gt_3),666).
orientation(br3gt_1,131,667). coords(br3gt_1,186,46,667).  visible(br3gt_1,667). happensAt(walking(br3gt_1),667). orientation(br3gt_3,150,667). coords(br3gt_3,4,89,667).  visible(br3gt_3,667). happensAt(walking(br3gt_3),667).
orientation(br3gt_1,131,668). coords(br3gt_1,186,46,668).  visible(br3gt_1,668). happensAt(walking(br3gt_1),668). orientation(br3gt_3,150,668). coords(br3gt_3,4,89,668).  visible(br3gt_3,668). happensAt(walking(br3gt_3),668).
orientation(br3gt_1,131,669). coords(br3gt_1,186,46,669).  visible(br3gt_1,669). happensAt(walking(br3gt_1),669). orientation(br3gt_3,150,669). coords(br3gt_3,4,89,669).  visible(br3gt_3,669). happensAt(walking(br3gt_3),669).
orientation(br3gt_1,131,670). coords(br3gt_1,186,46,670).  visible(br3gt_1,670). happensAt(walking(br3gt_1),670). orientation(br3gt_3,150,670). coords(br3gt_3,4,89,670).  visible(br3gt_3,670). happensAt(walking(br3gt_3),670).
orientation(br3gt_1,131,671). coords(br3gt_1,186,46,671).  visible(br3gt_1,671). happensAt(walking(br3gt_1),671). orientation(br3gt_3,150,671). coords(br3gt_3,4,88,671).  visible(br3gt_3,671). happensAt(walking(br3gt_3),671).
orientation(br3gt_1,131,672). coords(br3gt_1,186,47,672).  visible(br3gt_1,672). happensAt(walking(br3gt_1),672). orientation(br3gt_3,150,672). coords(br3gt_3,4,88,672).  visible(br3gt_3,672). happensAt(walking(br3gt_3),672).
orientation(br3gt_1,131,673). coords(br3gt_1,186,47,673).  visible(br3gt_1,673). happensAt(walking(br3gt_1),673). orientation(br3gt_3,150,673). coords(br3gt_3,5,86,673).  visible(br3gt_3,673). happensAt(walking(br3gt_3),673).
orientation(br3gt_1,131,674). coords(br3gt_1,186,47,674).  visible(br3gt_1,674). happensAt(walking(br3gt_1),674). orientation(br3gt_3,150,674). coords(br3gt_3,5,86,674).  visible(br3gt_3,674). happensAt(walking(br3gt_3),674).
orientation(br3gt_1,131,675). coords(br3gt_1,186,47,675).  visible(br3gt_1,675). happensAt(walking(br3gt_1),675). orientation(br3gt_3,150,675). coords(br3gt_3,6,83,675).  visible(br3gt_3,675). happensAt(walking(br3gt_3),675).
orientation(br3gt_1,131,676). coords(br3gt_1,186,47,676).  visible(br3gt_1,676). happensAt(walking(br3gt_1),676). orientation(br3gt_3,150,676). coords(br3gt_3,6,81,676).  visible(br3gt_3,676). happensAt(walking(br3gt_3),676).
orientation(br3gt_1,131,677). coords(br3gt_1,186,47,677).  visible(br3gt_1,677). happensAt(walking(br3gt_1),677). orientation(br3gt_3,150,677). coords(br3gt_3,6,81,677).  visible(br3gt_3,677). happensAt(walking(br3gt_3),677).
orientation(br3gt_1,131,678). coords(br3gt_1,186,47,678).  visible(br3gt_1,678). happensAt(walking(br3gt_1),678). orientation(br3gt_3,150,678). coords(br3gt_3,6,79,678).  visible(br3gt_3,678). happensAt(walking(br3gt_3),678).
#show holdsAt/2.