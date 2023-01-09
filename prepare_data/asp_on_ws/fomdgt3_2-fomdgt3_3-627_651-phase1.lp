
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

	orientation(fomdgt3_2,130,627). coords(fomdgt3_2,263,156,627).  visible(fomdgt3_2,627). happensAt(walking(fomdgt3_2),627). orientation(fomdgt3_3,159,627). coords(fomdgt3_3,71,95,627).  visible(fomdgt3_3,627). happensAt(walking(fomdgt3_3),627).
orientation(fomdgt3_2,130,628). coords(fomdgt3_2,264,157,628).  visible(fomdgt3_2,628). happensAt(walking(fomdgt3_2),628). orientation(fomdgt3_3,159,628). coords(fomdgt3_3,74,95,628).  visible(fomdgt3_3,628). happensAt(walking(fomdgt3_3),628).
orientation(fomdgt3_2,130,629). coords(fomdgt3_2,264,158,629).  visible(fomdgt3_2,629). happensAt(walking(fomdgt3_2),629). orientation(fomdgt3_3,159,629). coords(fomdgt3_3,74,95,629).  visible(fomdgt3_3,629). happensAt(walking(fomdgt3_3),629).
orientation(fomdgt3_2,130,630). coords(fomdgt3_2,265,159,630).  visible(fomdgt3_2,630). happensAt(walking(fomdgt3_2),630). orientation(fomdgt3_3,159,630). coords(fomdgt3_3,74,95,630).  visible(fomdgt3_3,630). happensAt(walking(fomdgt3_3),630).
orientation(fomdgt3_2,130,631). coords(fomdgt3_2,265,160,631).  visible(fomdgt3_2,631). happensAt(walking(fomdgt3_2),631). orientation(fomdgt3_3,159,631). coords(fomdgt3_3,75,95,631).  visible(fomdgt3_3,631). happensAt(walking(fomdgt3_3),631).
orientation(fomdgt3_2,130,632). coords(fomdgt3_2,265,161,632).  visible(fomdgt3_2,632). happensAt(walking(fomdgt3_2),632). orientation(fomdgt3_3,159,632). coords(fomdgt3_3,75,95,632).  visible(fomdgt3_3,632). happensAt(walking(fomdgt3_3),632).
orientation(fomdgt3_2,130,633). coords(fomdgt3_2,265,162,633).  visible(fomdgt3_2,633). happensAt(walking(fomdgt3_2),633). orientation(fomdgt3_3,159,633). coords(fomdgt3_3,75,95,633).  visible(fomdgt3_3,633). happensAt(walking(fomdgt3_3),633).
orientation(fomdgt3_2,130,634). coords(fomdgt3_2,267,163,634).  visible(fomdgt3_2,634). happensAt(walking(fomdgt3_2),634). orientation(fomdgt3_3,159,634). coords(fomdgt3_3,75,95,634).  visible(fomdgt3_3,634). happensAt(walking(fomdgt3_3),634).
orientation(fomdgt3_2,130,635). coords(fomdgt3_2,269,166,635).  visible(fomdgt3_2,635). happensAt(walking(fomdgt3_2),635). orientation(fomdgt3_3,159,635). coords(fomdgt3_3,75,95,635).  visible(fomdgt3_3,635). happensAt(walking(fomdgt3_3),635).
orientation(fomdgt3_2,130,636). coords(fomdgt3_2,271,171,636).  visible(fomdgt3_2,636). happensAt(walking(fomdgt3_2),636). orientation(fomdgt3_3,159,636). coords(fomdgt3_3,75,95,636).  visible(fomdgt3_3,636). happensAt(walking(fomdgt3_3),636).
orientation(fomdgt3_2,130,637). coords(fomdgt3_2,271,174,637).  visible(fomdgt3_2,637). happensAt(walking(fomdgt3_2),637). orientation(fomdgt3_3,159,637). coords(fomdgt3_3,78,95,637).  visible(fomdgt3_3,637). happensAt(walking(fomdgt3_3),637).
orientation(fomdgt3_2,130,638). coords(fomdgt3_2,271,177,638).  visible(fomdgt3_2,638). happensAt(walking(fomdgt3_2),638). orientation(fomdgt3_3,159,638). coords(fomdgt3_3,78,95,638).  visible(fomdgt3_3,638). happensAt(walking(fomdgt3_3),638).
orientation(fomdgt3_2,130,639). coords(fomdgt3_2,272,180,639).  visible(fomdgt3_2,639). happensAt(walking(fomdgt3_2),639). orientation(fomdgt3_3,159,639). coords(fomdgt3_3,80,95,639).  visible(fomdgt3_3,639). happensAt(walking(fomdgt3_3),639).
orientation(fomdgt3_2,130,640). coords(fomdgt3_2,273,181,640).  visible(fomdgt3_2,640). happensAt(walking(fomdgt3_2),640). orientation(fomdgt3_3,159,640). coords(fomdgt3_3,80,95,640).  visible(fomdgt3_3,640). happensAt(walking(fomdgt3_3),640).
orientation(fomdgt3_2,130,641). coords(fomdgt3_2,273,182,641).  visible(fomdgt3_2,641). happensAt(walking(fomdgt3_2),641). orientation(fomdgt3_3,159,641). coords(fomdgt3_3,81,95,641).  visible(fomdgt3_3,641). happensAt(walking(fomdgt3_3),641).
orientation(fomdgt3_2,130,642). coords(fomdgt3_2,273,183,642).  visible(fomdgt3_2,642). happensAt(walking(fomdgt3_2),642). orientation(fomdgt3_3,159,642). coords(fomdgt3_3,81,95,642).  visible(fomdgt3_3,642). happensAt(walking(fomdgt3_3),642).
orientation(fomdgt3_2,130,643). coords(fomdgt3_2,274,184,643).  visible(fomdgt3_2,643). happensAt(walking(fomdgt3_2),643). orientation(fomdgt3_3,159,643). coords(fomdgt3_3,83,95,643).  visible(fomdgt3_3,643). happensAt(walking(fomdgt3_3),643).
orientation(fomdgt3_2,130,644). coords(fomdgt3_2,274,184,644).  visible(fomdgt3_2,644). happensAt(walking(fomdgt3_2),644). orientation(fomdgt3_3,159,644). coords(fomdgt3_3,83,95,644).  visible(fomdgt3_3,644). happensAt(walking(fomdgt3_3),644).
orientation(fomdgt3_2,130,645). coords(fomdgt3_2,274,184,645).  visible(fomdgt3_2,645). happensAt(walking(fomdgt3_2),645). orientation(fomdgt3_3,159,645). coords(fomdgt3_3,83,95,645).  visible(fomdgt3_3,645). happensAt(walking(fomdgt3_3),645).
orientation(fomdgt3_2,130,646). coords(fomdgt3_2,274,187,646).  visible(fomdgt3_2,646). happensAt(walking(fomdgt3_2),646). orientation(fomdgt3_3,159,646). coords(fomdgt3_3,83,95,646).  visible(fomdgt3_3,646). happensAt(walking(fomdgt3_3),646).
orientation(fomdgt3_2,130,647). coords(fomdgt3_2,274,188,647).  visible(fomdgt3_2,647). happensAt(walking(fomdgt3_2),647). orientation(fomdgt3_3,159,647). coords(fomdgt3_3,83,95,647).  visible(fomdgt3_3,647). happensAt(walking(fomdgt3_3),647).
orientation(fomdgt3_2,130,648). coords(fomdgt3_2,274,190,648).  visible(fomdgt3_2,648). happensAt(walking(fomdgt3_2),648). orientation(fomdgt3_3,159,648). coords(fomdgt3_3,83,95,648).  visible(fomdgt3_3,648). happensAt(walking(fomdgt3_3),648).
orientation(fomdgt3_2,130,649). coords(fomdgt3_2,274,191,649).  visible(fomdgt3_2,649). happensAt(walking(fomdgt3_2),649). orientation(fomdgt3_3,159,649). coords(fomdgt3_3,84,96,649).  visible(fomdgt3_3,649). happensAt(walking(fomdgt3_3),649).
orientation(fomdgt3_2,130,650). coords(fomdgt3_2,277,194,650).  visible(fomdgt3_2,650). happensAt(walking(fomdgt3_2),650). orientation(fomdgt3_3,159,650). coords(fomdgt3_3,85,96,650).  visible(fomdgt3_3,650). happensAt(walking(fomdgt3_3),650).
orientation(fomdgt3_2,130,651). coords(fomdgt3_2,280,197,651).  visible(fomdgt3_2,651). happensAt(walking(fomdgt3_2),651). orientation(fomdgt3_3,159,651). coords(fomdgt3_3,86,97,651).  visible(fomdgt3_3,651). happensAt(walking(fomdgt3_3),651).
#show holdsAt/2.