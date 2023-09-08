
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

	orientation(br3gt_1,131,654). coords(br3gt_1,184,47,654).  visible(br3gt_1,654). happensAt(active(br3gt_1),654). orientation(br3gt_2,150,654). coords(br3gt_2,6,77,654).  visible(br3gt_2,654). happensAt(walking(br3gt_2),654).
orientation(br3gt_1,131,655). coords(br3gt_1,184,47,655).  visible(br3gt_1,655). happensAt(active(br3gt_1),655). orientation(br3gt_2,148,655). coords(br3gt_2,6,77,655).  visible(br3gt_2,655). happensAt(walking(br3gt_2),655).
orientation(br3gt_1,131,656). coords(br3gt_1,184,47,656).  visible(br3gt_1,656). happensAt(active(br3gt_1),656). orientation(br3gt_2,148,656). coords(br3gt_2,6,77,656).  visible(br3gt_2,656). happensAt(walking(br3gt_2),656).
orientation(br3gt_1,131,657). coords(br3gt_1,184,47,657).  visible(br3gt_1,657). happensAt(active(br3gt_1),657). orientation(br3gt_2,148,657). coords(br3gt_2,6,77,657).  visible(br3gt_2,657). happensAt(walking(br3gt_2),657).
orientation(br3gt_1,131,658). coords(br3gt_1,184,47,658).  visible(br3gt_1,658). happensAt(active(br3gt_1),658). orientation(br3gt_2,148,658). coords(br3gt_2,7,74,658).  visible(br3gt_2,658). happensAt(walking(br3gt_2),658).
orientation(br3gt_1,131,659). coords(br3gt_1,184,47,659).  visible(br3gt_1,659). happensAt(active(br3gt_1),659). orientation(br3gt_2,148,659). coords(br3gt_2,7,73,659).  visible(br3gt_2,659). happensAt(walking(br3gt_2),659).
orientation(br3gt_1,131,660). coords(br3gt_1,184,47,660).  visible(br3gt_1,660). happensAt(walking(br3gt_1),660). orientation(br3gt_2,148,660). coords(br3gt_2,7,73,660).  visible(br3gt_2,660). happensAt(walking(br3gt_2),660).
orientation(br3gt_1,131,661). coords(br3gt_1,184,47,661).  visible(br3gt_1,661). happensAt(walking(br3gt_1),661). orientation(br3gt_2,148,661). coords(br3gt_2,7,72,661).  visible(br3gt_2,661). happensAt(walking(br3gt_2),661).
orientation(br3gt_1,131,662). coords(br3gt_1,184,46,662).  visible(br3gt_1,662). happensAt(walking(br3gt_1),662). orientation(br3gt_2,148,662). coords(br3gt_2,7,71,662).  visible(br3gt_2,662). happensAt(walking(br3gt_2),662).
orientation(br3gt_1,131,663). coords(br3gt_1,184,46,663).  visible(br3gt_1,663). happensAt(walking(br3gt_1),663). orientation(br3gt_2,148,663). coords(br3gt_2,7,69,663).  visible(br3gt_2,663). happensAt(walking(br3gt_2),663).
orientation(br3gt_1,131,664). coords(br3gt_1,185,46,664).  visible(br3gt_1,664). happensAt(walking(br3gt_1),664). orientation(br3gt_2,148,664). coords(br3gt_2,7,69,664).  visible(br3gt_2,664). happensAt(walking(br3gt_2),664).
orientation(br3gt_1,131,665). coords(br3gt_1,185,46,665).  visible(br3gt_1,665). happensAt(walking(br3gt_1),665). orientation(br3gt_2,148,665). coords(br3gt_2,8,68,665).  visible(br3gt_2,665). happensAt(walking(br3gt_2),665).
orientation(br3gt_1,131,666). coords(br3gt_1,185,46,666).  visible(br3gt_1,666). happensAt(walking(br3gt_1),666). orientation(br3gt_2,148,666). coords(br3gt_2,8,68,666).  visible(br3gt_2,666). happensAt(walking(br3gt_2),666).
orientation(br3gt_1,131,667). coords(br3gt_1,186,46,667).  visible(br3gt_1,667). happensAt(walking(br3gt_1),667). orientation(br3gt_2,148,667). coords(br3gt_2,9,67,667).  visible(br3gt_2,667). happensAt(walking(br3gt_2),667).
orientation(br3gt_1,131,668). coords(br3gt_1,186,46,668).  visible(br3gt_1,668). happensAt(walking(br3gt_1),668). orientation(br3gt_2,148,668). coords(br3gt_2,9,66,668).  visible(br3gt_2,668). happensAt(walking(br3gt_2),668).
orientation(br3gt_1,131,669). coords(br3gt_1,186,46,669).  visible(br3gt_1,669). happensAt(walking(br3gt_1),669). orientation(br3gt_2,148,669). coords(br3gt_2,9,65,669).  visible(br3gt_2,669). happensAt(walking(br3gt_2),669).
orientation(br3gt_1,131,670). coords(br3gt_1,186,46,670).  visible(br3gt_1,670). happensAt(walking(br3gt_1),670). orientation(br3gt_2,148,670). coords(br3gt_2,9,64,670).  visible(br3gt_2,670). happensAt(walking(br3gt_2),670).
orientation(br3gt_1,131,671). coords(br3gt_1,186,46,671).  visible(br3gt_1,671). happensAt(walking(br3gt_1),671). orientation(br3gt_2,148,671). coords(br3gt_2,9,63,671).  visible(br3gt_2,671). happensAt(walking(br3gt_2),671).
orientation(br3gt_1,131,672). coords(br3gt_1,186,47,672).  visible(br3gt_1,672). happensAt(walking(br3gt_1),672). orientation(br3gt_2,148,672). coords(br3gt_2,9,63,672).  visible(br3gt_2,672). happensAt(walking(br3gt_2),672).
orientation(br3gt_1,131,673). coords(br3gt_1,186,47,673).  visible(br3gt_1,673). happensAt(walking(br3gt_1),673). orientation(br3gt_2,148,673). coords(br3gt_2,9,62,673).  visible(br3gt_2,673). happensAt(walking(br3gt_2),673).
orientation(br3gt_1,131,674). coords(br3gt_1,186,47,674).  visible(br3gt_1,674). happensAt(walking(br3gt_1),674). orientation(br3gt_2,148,674). coords(br3gt_2,10,61,674).  visible(br3gt_2,674). happensAt(walking(br3gt_2),674).
orientation(br3gt_1,131,675). coords(br3gt_1,186,47,675).  visible(br3gt_1,675). happensAt(walking(br3gt_1),675). orientation(br3gt_2,148,675). coords(br3gt_2,11,61,675).  visible(br3gt_2,675). happensAt(walking(br3gt_2),675).
orientation(br3gt_1,131,676). coords(br3gt_1,186,47,676).  visible(br3gt_1,676). happensAt(walking(br3gt_1),676). orientation(br3gt_2,141,676). coords(br3gt_2,11,60,676).  visible(br3gt_2,676). happensAt(walking(br3gt_2),676).
orientation(br3gt_1,131,677). coords(br3gt_1,186,47,677).  visible(br3gt_1,677). happensAt(walking(br3gt_1),677). orientation(br3gt_2,141,677). coords(br3gt_2,11,60,677).  visible(br3gt_2,677). happensAt(walking(br3gt_2),677).
orientation(br3gt_1,131,678). coords(br3gt_1,186,47,678).  visible(br3gt_1,678). happensAt(walking(br3gt_1),678). orientation(br3gt_2,138,678). coords(br3gt_2,12,59,678).  visible(br3gt_2,678). happensAt(walking(br3gt_2),678).
#show holdsAt/2.