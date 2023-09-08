
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

	orientation(mwt2gt_1,153,649). coords(mwt2gt_1,22,139,649).  visible(mwt2gt_1,649). happensAt(walking(mwt2gt_1),649). orientation(mwt2gt_2,152,649). coords(mwt2gt_2,22,141,649).  visible(mwt2gt_2,649). happensAt(walking(mwt2gt_2),649).
orientation(mwt2gt_1,153,650). coords(mwt2gt_1,22,139,650).  visible(mwt2gt_1,650). happensAt(walking(mwt2gt_1),650). orientation(mwt2gt_2,152,650). coords(mwt2gt_2,22,141,650).  visible(mwt2gt_2,650). happensAt(walking(mwt2gt_2),650).
orientation(mwt2gt_1,153,651). coords(mwt2gt_1,22,137,651).  visible(mwt2gt_1,651). happensAt(walking(mwt2gt_1),651). orientation(mwt2gt_2,152,651). coords(mwt2gt_2,22,141,651).  visible(mwt2gt_2,651). happensAt(walking(mwt2gt_2),651).
orientation(mwt2gt_1,153,652). coords(mwt2gt_1,22,137,652).  visible(mwt2gt_1,652). happensAt(walking(mwt2gt_1),652). orientation(mwt2gt_2,152,652). coords(mwt2gt_2,22,141,652).  visible(mwt2gt_2,652). happensAt(walking(mwt2gt_2),652).
orientation(mwt2gt_1,153,653). coords(mwt2gt_1,22,136,653).  visible(mwt2gt_1,653). happensAt(walking(mwt2gt_1),653). orientation(mwt2gt_2,152,653). coords(mwt2gt_2,22,141,653).  visible(mwt2gt_2,653). happensAt(walking(mwt2gt_2),653).
orientation(mwt2gt_1,153,654). coords(mwt2gt_1,22,136,654).  visible(mwt2gt_1,654). happensAt(walking(mwt2gt_1),654). orientation(mwt2gt_2,152,654). coords(mwt2gt_2,22,140,654).  visible(mwt2gt_2,654). happensAt(walking(mwt2gt_2),654).
orientation(mwt2gt_1,153,655). coords(mwt2gt_1,22,136,655).  visible(mwt2gt_1,655). happensAt(walking(mwt2gt_1),655). orientation(mwt2gt_2,152,655). coords(mwt2gt_2,22,140,655).  visible(mwt2gt_2,655). happensAt(walking(mwt2gt_2),655).
orientation(mwt2gt_1,153,656). coords(mwt2gt_1,22,135,656).  visible(mwt2gt_1,656). happensAt(walking(mwt2gt_1),656). orientation(mwt2gt_2,152,656). coords(mwt2gt_2,22,140,656).  visible(mwt2gt_2,656). happensAt(walking(mwt2gt_2),656).
orientation(mwt2gt_1,153,657). coords(mwt2gt_1,23,135,657).  visible(mwt2gt_1,657). happensAt(walking(mwt2gt_1),657). orientation(mwt2gt_2,152,657). coords(mwt2gt_2,22,140,657).  visible(mwt2gt_2,657). happensAt(walking(mwt2gt_2),657).
orientation(mwt2gt_1,153,658). coords(mwt2gt_1,23,135,658).  visible(mwt2gt_1,658). happensAt(walking(mwt2gt_1),658). orientation(mwt2gt_2,152,658). coords(mwt2gt_2,22,140,658).  visible(mwt2gt_2,658). happensAt(walking(mwt2gt_2),658).
orientation(mwt2gt_1,153,659). coords(mwt2gt_1,23,134,659).  visible(mwt2gt_1,659). happensAt(walking(mwt2gt_1),659). orientation(mwt2gt_2,152,659). coords(mwt2gt_2,22,140,659).  visible(mwt2gt_2,659). happensAt(walking(mwt2gt_2),659).
orientation(mwt2gt_1,153,660). coords(mwt2gt_1,23,134,660).  visible(mwt2gt_1,660). happensAt(walking(mwt2gt_1),660). orientation(mwt2gt_2,152,660). coords(mwt2gt_2,22,138,660).  visible(mwt2gt_2,660). happensAt(walking(mwt2gt_2),660).
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
#show holdsAt/2.