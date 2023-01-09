
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

	orientation(bww1gt_1,173,650). coords(bww1gt_1,102,257,650).  visible(bww1gt_1,650). happensAt(walking(bww1gt_1),650). orientation(bww1gt_2,169,650). coords(bww1gt_2,192,226,650).  visible(bww1gt_2,650). happensAt(walking(bww1gt_2),650).
orientation(bww1gt_1,173,651). coords(bww1gt_1,102,257,651).  visible(bww1gt_1,651). happensAt(walking(bww1gt_1),651). orientation(bww1gt_2,169,651). coords(bww1gt_2,193,228,651).  visible(bww1gt_2,651). happensAt(walking(bww1gt_2),651).
orientation(bww1gt_1,173,652). coords(bww1gt_1,102,257,652).  visible(bww1gt_1,652). happensAt(walking(bww1gt_1),652). orientation(bww1gt_2,169,652). coords(bww1gt_2,193,230,652).  visible(bww1gt_2,652). happensAt(walking(bww1gt_2),652).
orientation(bww1gt_1,173,653). coords(bww1gt_1,102,257,653).  visible(bww1gt_1,653). happensAt(walking(bww1gt_1),653). orientation(bww1gt_2,169,653). coords(bww1gt_2,194,232,653).  visible(bww1gt_2,653). happensAt(walking(bww1gt_2),653).
orientation(bww1gt_1,173,654). coords(bww1gt_1,102,257,654).  visible(bww1gt_1,654). happensAt(walking(bww1gt_1),654). orientation(bww1gt_2,169,654). coords(bww1gt_2,196,235,654).  visible(bww1gt_2,654). happensAt(walking(bww1gt_2),654).
orientation(bww1gt_1,173,655). coords(bww1gt_1,102,257,655).  visible(bww1gt_1,655). happensAt(walking(bww1gt_1),655). orientation(bww1gt_2,169,655). coords(bww1gt_2,198,237,655).  visible(bww1gt_2,655). happensAt(walking(bww1gt_2),655).
orientation(bww1gt_1,173,656). coords(bww1gt_1,102,257,656).  visible(bww1gt_1,656). happensAt(inactive(bww1gt_1),656). orientation(bww1gt_2,169,656). coords(bww1gt_2,199,239,656).  visible(bww1gt_2,656). happensAt(walking(bww1gt_2),656).
orientation(bww1gt_1,173,657). coords(bww1gt_1,102,257,657).  visible(bww1gt_1,657). happensAt(inactive(bww1gt_1),657). orientation(bww1gt_2,169,657). coords(bww1gt_2,200,239,657).  visible(bww1gt_2,657). happensAt(walking(bww1gt_2),657).
orientation(bww1gt_1,173,658). coords(bww1gt_1,102,257,658).  visible(bww1gt_1,658). happensAt(inactive(bww1gt_1),658). orientation(bww1gt_2,169,658). coords(bww1gt_2,201,240,658).  visible(bww1gt_2,658). happensAt(walking(bww1gt_2),658).
orientation(bww1gt_1,173,659). coords(bww1gt_1,102,257,659).  visible(bww1gt_1,659). happensAt(inactive(bww1gt_1),659). orientation(bww1gt_2,169,659). coords(bww1gt_2,203,240,659).  visible(bww1gt_2,659). happensAt(walking(bww1gt_2),659).
orientation(bww1gt_1,173,660). coords(bww1gt_1,102,257,660).  visible(bww1gt_1,660). happensAt(inactive(bww1gt_1),660). orientation(bww1gt_2,169,660). coords(bww1gt_2,205,241,660).  visible(bww1gt_2,660). happensAt(walking(bww1gt_2),660).
orientation(bww1gt_1,173,661). coords(bww1gt_1,102,257,661).  visible(bww1gt_1,661). happensAt(inactive(bww1gt_1),661). orientation(bww1gt_2,169,661). coords(bww1gt_2,205,242,661).  visible(bww1gt_2,661). happensAt(walking(bww1gt_2),661).
orientation(bww1gt_1,173,662). coords(bww1gt_1,102,257,662).  visible(bww1gt_1,662). happensAt(inactive(bww1gt_1),662). orientation(bww1gt_2,169,662). coords(bww1gt_2,205,244,662).  visible(bww1gt_2,662). happensAt(walking(bww1gt_2),662).
orientation(bww1gt_1,173,663). coords(bww1gt_1,102,257,663).  visible(bww1gt_1,663). happensAt(inactive(bww1gt_1),663). orientation(bww1gt_2,4,663). coords(bww1gt_2,205,246,663).  visible(bww1gt_2,663). happensAt(walking(bww1gt_2),663).
orientation(bww1gt_1,173,664). coords(bww1gt_1,102,257,664).  visible(bww1gt_1,664). happensAt(inactive(bww1gt_1),664). orientation(bww1gt_2,179,664). coords(bww1gt_2,206,249,664).  visible(bww1gt_2,664). happensAt(walking(bww1gt_2),664).
orientation(bww1gt_1,173,665). coords(bww1gt_1,102,257,665).  visible(bww1gt_1,665). happensAt(inactive(bww1gt_1),665). orientation(bww1gt_2,2,665). coords(bww1gt_2,206,252,665).  visible(bww1gt_2,665). happensAt(walking(bww1gt_2),665).
orientation(bww1gt_1,173,666). coords(bww1gt_1,102,257,666).  visible(bww1gt_1,666). happensAt(inactive(bww1gt_1),666). orientation(bww1gt_2,176,666). coords(bww1gt_2,208,255,666).  visible(bww1gt_2,666). happensAt(walking(bww1gt_2),666).
orientation(bww1gt_1,173,667). coords(bww1gt_1,102,257,667).  visible(bww1gt_1,667). happensAt(inactive(bww1gt_1),667). orientation(bww1gt_2,3,667). coords(bww1gt_2,211,258,667).  visible(bww1gt_2,667). happensAt(walking(bww1gt_2),667).
orientation(bww1gt_1,173,668). coords(bww1gt_1,102,257,668).  visible(bww1gt_1,668). happensAt(inactive(bww1gt_1),668). orientation(bww1gt_2,2,668). coords(bww1gt_2,214,260,668).  visible(bww1gt_2,668). happensAt(walking(bww1gt_2),668).
orientation(bww1gt_1,173,669). coords(bww1gt_1,102,257,669).  visible(bww1gt_1,669). happensAt(inactive(bww1gt_1),669). orientation(bww1gt_2,10,669). coords(bww1gt_2,215,262,669).  visible(bww1gt_2,669). happensAt(walking(bww1gt_2),669).
orientation(bww1gt_1,173,670). coords(bww1gt_1,102,257,670).  visible(bww1gt_1,670). happensAt(inactive(bww1gt_1),670). orientation(bww1gt_2,7,670). coords(bww1gt_2,216,263,670).  visible(bww1gt_2,670). happensAt(walking(bww1gt_2),670).
orientation(bww1gt_1,173,671). coords(bww1gt_1,102,257,671).  visible(bww1gt_1,671). happensAt(inactive(bww1gt_1),671). orientation(bww1gt_2,4,671). coords(bww1gt_2,217,264,671).  visible(bww1gt_2,671). happensAt(walking(bww1gt_2),671).
orientation(bww1gt_1,173,672). coords(bww1gt_1,102,257,672).  visible(bww1gt_1,672). happensAt(inactive(bww1gt_1),672). orientation(bww1gt_2,10,672). coords(bww1gt_2,218,265,672).  visible(bww1gt_2,672). happensAt(walking(bww1gt_2),672).
orientation(bww1gt_1,173,673). coords(bww1gt_1,102,257,673).  visible(bww1gt_1,673). happensAt(inactive(bww1gt_1),673). orientation(bww1gt_2,14,673). coords(bww1gt_2,219,267,673).  visible(bww1gt_2,673). happensAt(walking(bww1gt_2),673).
orientation(bww1gt_1,173,674). coords(bww1gt_1,102,257,674).  visible(bww1gt_1,674). happensAt(inactive(bww1gt_1),674). orientation(bww1gt_2,8,674). coords(bww1gt_2,219,268,674).  visible(bww1gt_2,674). happensAt(walking(bww1gt_2),674).
#show holdsAt/2.