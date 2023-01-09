
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

	orientation(lbgt_3,61,631). coords(lbgt_3,146,162,631).  visible(lbgt_3,631). happensAt(inactive(lbgt_3),631). orientation(lbgt_4,164,631). coords(lbgt_4,113,167,631).  visible(lbgt_4,631). happensAt(walking(lbgt_4),631).
orientation(lbgt_3,61,632). coords(lbgt_3,146,162,632).  visible(lbgt_3,632). happensAt(inactive(lbgt_3),632). orientation(lbgt_4,160,632). coords(lbgt_4,113,169,632).  visible(lbgt_4,632). happensAt(walking(lbgt_4),632).
orientation(lbgt_3,61,633). coords(lbgt_3,146,162,633).  visible(lbgt_3,633). happensAt(inactive(lbgt_3),633). orientation(lbgt_4,161,633). coords(lbgt_4,113,170,633).  visible(lbgt_4,633). happensAt(walking(lbgt_4),633).
orientation(lbgt_3,61,634). coords(lbgt_3,146,162,634).  visible(lbgt_3,634). happensAt(inactive(lbgt_3),634). orientation(lbgt_4,148,634). coords(lbgt_4,113,172,634).  visible(lbgt_4,634). happensAt(walking(lbgt_4),634).
orientation(lbgt_3,61,635). coords(lbgt_3,146,162,635).  visible(lbgt_3,635). happensAt(inactive(lbgt_3),635). orientation(lbgt_4,148,635). coords(lbgt_4,114,174,635).  visible(lbgt_4,635). happensAt(walking(lbgt_4),635).
orientation(lbgt_3,61,636). coords(lbgt_3,146,162,636).  visible(lbgt_3,636). happensAt(inactive(lbgt_3),636). orientation(lbgt_4,158,636). coords(lbgt_4,114,174,636).  visible(lbgt_4,636). happensAt(walking(lbgt_4),636).
orientation(lbgt_3,61,637). coords(lbgt_3,146,162,637).  visible(lbgt_3,637). happensAt(inactive(lbgt_3),637). orientation(lbgt_4,159,637). coords(lbgt_4,115,175,637).  visible(lbgt_4,637). happensAt(walking(lbgt_4),637).
orientation(lbgt_3,61,638). coords(lbgt_3,146,162,638).  visible(lbgt_3,638). happensAt(inactive(lbgt_3),638). orientation(lbgt_4,161,638). coords(lbgt_4,116,176,638).  visible(lbgt_4,638). happensAt(walking(lbgt_4),638).
orientation(lbgt_3,61,639). coords(lbgt_3,146,162,639).  visible(lbgt_3,639). happensAt(inactive(lbgt_3),639). orientation(lbgt_4,160,639). coords(lbgt_4,116,176,639).  visible(lbgt_4,639). happensAt(walking(lbgt_4),639).
orientation(lbgt_3,61,640). coords(lbgt_3,146,162,640).  visible(lbgt_3,640). happensAt(inactive(lbgt_3),640). orientation(lbgt_4,163,640). coords(lbgt_4,116,177,640).  visible(lbgt_4,640). happensAt(walking(lbgt_4),640).
orientation(lbgt_3,61,641). coords(lbgt_3,146,162,641).  visible(lbgt_3,641). happensAt(inactive(lbgt_3),641). orientation(lbgt_4,159,641). coords(lbgt_4,116,177,641).  visible(lbgt_4,641). happensAt(walking(lbgt_4),641).
orientation(lbgt_3,61,642). coords(lbgt_3,146,162,642).  visible(lbgt_3,642). happensAt(inactive(lbgt_3),642). orientation(lbgt_4,158,642). coords(lbgt_4,116,177,642).  visible(lbgt_4,642). happensAt(walking(lbgt_4),642).
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
#show holdsAt/2.