
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

	orientation(lbgt_3,61,643). coords(lbgt_3,146,162,643).  visible(lbgt_3,643). happensAt(inactive(lbgt_3),643). orientation(lbgt_4,159,643). coords(lbgt_4,116,177,643).  visible(lbgt_4,643). happensAt(walking(lbgt_4),643).
orientation(lbgt_3,61,644). coords(lbgt_3,146,162,644).  visible(lbgt_3,644). happensAt(inactive(lbgt_3),644). orientation(lbgt_4,167,644). coords(lbgt_4,116,177,644).  visible(lbgt_4,644). happensAt(walking(lbgt_4),644).
orientation(lbgt_3,61,645). coords(lbgt_3,146,162,645).  visible(lbgt_3,645). happensAt(inactive(lbgt_3),645). orientation(lbgt_4,163,645). coords(lbgt_4,116,179,645).  visible(lbgt_4,645). happensAt(walking(lbgt_4),645).
orientation(lbgt_3,61,646). coords(lbgt_3,146,162,646).  visible(lbgt_3,646). happensAt(inactive(lbgt_3),646). orientation(lbgt_4,164,646). coords(lbgt_4,117,180,646).  visible(lbgt_4,646). happensAt(walking(lbgt_4),646).
orientation(lbgt_3,61,647). coords(lbgt_3,146,162,647).  visible(lbgt_3,647). happensAt(inactive(lbgt_3),647). orientation(lbgt_4,159,647). coords(lbgt_4,120,182,647).  visible(lbgt_4,647). happensAt(walking(lbgt_4),647).
orientation(lbgt_3,61,648). coords(lbgt_3,146,162,648).  visible(lbgt_3,648). happensAt(inactive(lbgt_3),648). orientation(lbgt_4,160,648). coords(lbgt_4,120,183,648).  visible(lbgt_4,648). happensAt(walking(lbgt_4),648).
orientation(lbgt_3,61,649). coords(lbgt_3,146,162,649).  visible(lbgt_3,649). happensAt(inactive(lbgt_3),649). orientation(lbgt_4,157,649). coords(lbgt_4,120,184,649).  visible(lbgt_4,649). happensAt(walking(lbgt_4),649).
orientation(lbgt_3,61,650). coords(lbgt_3,146,162,650).  visible(lbgt_3,650). happensAt(inactive(lbgt_3),650). orientation(lbgt_4,160,650). coords(lbgt_4,122,184,650).  visible(lbgt_4,650). happensAt(walking(lbgt_4),650).
orientation(lbgt_3,61,651). coords(lbgt_3,146,162,651).  visible(lbgt_3,651). happensAt(inactive(lbgt_3),651). orientation(lbgt_4,155,651). coords(lbgt_4,123,185,651).  visible(lbgt_4,651). happensAt(walking(lbgt_4),651).
orientation(lbgt_3,61,652). coords(lbgt_3,146,162,652).  visible(lbgt_3,652). happensAt(inactive(lbgt_3),652). orientation(lbgt_4,155,652). coords(lbgt_4,124,185,652).  visible(lbgt_4,652). happensAt(walking(lbgt_4),652).
orientation(lbgt_3,61,653). coords(lbgt_3,146,162,653).  visible(lbgt_3,653). happensAt(inactive(lbgt_3),653). orientation(lbgt_4,161,653). coords(lbgt_4,124,185,653).  visible(lbgt_4,653). happensAt(walking(lbgt_4),653).
orientation(lbgt_3,61,654). coords(lbgt_3,146,162,654).  visible(lbgt_3,654). happensAt(inactive(lbgt_3),654). orientation(lbgt_4,161,654). coords(lbgt_4,125,186,654).  visible(lbgt_4,654). happensAt(walking(lbgt_4),654).
orientation(lbgt_3,61,655). coords(lbgt_3,146,162,655).  visible(lbgt_3,655). happensAt(inactive(lbgt_3),655). orientation(lbgt_4,161,655). coords(lbgt_4,125,186,655).  visible(lbgt_4,655). happensAt(walking(lbgt_4),655).
orientation(lbgt_3,61,656). coords(lbgt_3,146,162,656).  visible(lbgt_3,656). happensAt(inactive(lbgt_3),656). orientation(lbgt_4,161,656). coords(lbgt_4,125,186,656).  visible(lbgt_4,656). happensAt(walking(lbgt_4),656).
orientation(lbgt_3,61,657). coords(lbgt_3,146,162,657).  visible(lbgt_3,657). happensAt(inactive(lbgt_3),657). orientation(lbgt_4,161,657). coords(lbgt_4,125,186,657).  visible(lbgt_4,657). happensAt(walking(lbgt_4),657).
orientation(lbgt_3,61,658). coords(lbgt_3,146,162,658).  visible(lbgt_3,658). happensAt(inactive(lbgt_3),658). orientation(lbgt_4,161,658). coords(lbgt_4,127,187,658).  visible(lbgt_4,658). happensAt(walking(lbgt_4),658).
orientation(lbgt_3,61,659). coords(lbgt_3,146,162,659).  visible(lbgt_3,659). happensAt(inactive(lbgt_3),659). orientation(lbgt_4,161,659). coords(lbgt_4,127,187,659).  visible(lbgt_4,659). happensAt(walking(lbgt_4),659).
orientation(lbgt_3,61,660). coords(lbgt_3,146,162,660).  visible(lbgt_3,660). happensAt(inactive(lbgt_3),660). orientation(lbgt_4,161,660). coords(lbgt_4,127,187,660).  visible(lbgt_4,660). happensAt(walking(lbgt_4),660).
orientation(lbgt_3,61,661). coords(lbgt_3,146,162,661).  visible(lbgt_3,661). happensAt(inactive(lbgt_3),661). orientation(lbgt_4,161,661). coords(lbgt_4,127,187,661).  visible(lbgt_4,661). happensAt(walking(lbgt_4),661).
orientation(lbgt_3,61,662). coords(lbgt_3,146,162,662).  visible(lbgt_3,662). happensAt(inactive(lbgt_3),662). orientation(lbgt_4,161,662). coords(lbgt_4,127,187,662).  visible(lbgt_4,662). happensAt(walking(lbgt_4),662).
orientation(lbgt_3,61,663). coords(lbgt_3,146,162,663).  visible(lbgt_3,663). happensAt(inactive(lbgt_3),663). orientation(lbgt_4,161,663). coords(lbgt_4,127,187,663).  visible(lbgt_4,663). happensAt(walking(lbgt_4),663).
orientation(lbgt_3,61,664). coords(lbgt_3,146,162,664).  visible(lbgt_3,664). happensAt(inactive(lbgt_3),664). orientation(lbgt_4,161,664). coords(lbgt_4,128,187,664).  visible(lbgt_4,664). happensAt(walking(lbgt_4),664).
orientation(lbgt_3,61,665). coords(lbgt_3,146,162,665).  visible(lbgt_3,665). happensAt(inactive(lbgt_3),665). orientation(lbgt_4,161,665). coords(lbgt_4,128,187,665).  visible(lbgt_4,665). happensAt(walking(lbgt_4),665).
orientation(lbgt_3,61,666). coords(lbgt_3,146,162,666).  visible(lbgt_3,666). happensAt(inactive(lbgt_3),666). orientation(lbgt_4,161,666). coords(lbgt_4,130,188,666).  visible(lbgt_4,666). happensAt(active(lbgt_4),666).
orientation(lbgt_3,61,667). coords(lbgt_3,146,162,667).  visible(lbgt_3,667). happensAt(inactive(lbgt_3),667). orientation(lbgt_4,161,667). coords(lbgt_4,130,188,667).  visible(lbgt_4,667). happensAt(active(lbgt_4),667).
#show holdsAt/2.