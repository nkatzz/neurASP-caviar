
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

	orientation(mwt1gt_5,150,625). coords(mwt1gt_5,82,67,625).  visible(mwt1gt_5,625). happensAt(walking(mwt1gt_5),625). orientation(mwt1gt_6,150,625). coords(mwt1gt_6,78,65,625).  happensAt(appear(mwt1gt_6),625). happensAt(walking(mwt1gt_6),625).
orientation(mwt1gt_5,150,626). coords(mwt1gt_5,82,67,626).  visible(mwt1gt_5,626). happensAt(walking(mwt1gt_5),626). orientation(mwt1gt_6,150,626). coords(mwt1gt_6,78,65,626).  visible(mwt1gt_6,626). happensAt(walking(mwt1gt_6),626).
orientation(mwt1gt_5,150,627). coords(mwt1gt_5,82,67,627).  visible(mwt1gt_5,627). happensAt(walking(mwt1gt_5),627). orientation(mwt1gt_6,150,627). coords(mwt1gt_6,78,65,627).  visible(mwt1gt_6,627). happensAt(walking(mwt1gt_6),627).
orientation(mwt1gt_5,150,628). coords(mwt1gt_5,82,67,628).  visible(mwt1gt_5,628). happensAt(walking(mwt1gt_5),628). orientation(mwt1gt_6,150,628). coords(mwt1gt_6,78,65,628).  visible(mwt1gt_6,628). happensAt(walking(mwt1gt_6),628).
orientation(mwt1gt_5,150,629). coords(mwt1gt_5,82,67,629).  visible(mwt1gt_5,629). happensAt(walking(mwt1gt_5),629). orientation(mwt1gt_6,150,629). coords(mwt1gt_6,78,65,629).  visible(mwt1gt_6,629). happensAt(walking(mwt1gt_6),629).
orientation(mwt1gt_5,150,630). coords(mwt1gt_5,82,66,630).  visible(mwt1gt_5,630). happensAt(walking(mwt1gt_5),630). orientation(mwt1gt_6,150,630). coords(mwt1gt_6,78,71,630).  visible(mwt1gt_6,630). happensAt(walking(mwt1gt_6),630).
orientation(mwt1gt_5,150,631). coords(mwt1gt_5,82,66,631).  visible(mwt1gt_5,631). happensAt(walking(mwt1gt_5),631). orientation(mwt1gt_6,150,631). coords(mwt1gt_6,78,71,631).  visible(mwt1gt_6,631). happensAt(walking(mwt1gt_6),631).
orientation(mwt1gt_5,150,632). coords(mwt1gt_5,82,66,632).  visible(mwt1gt_5,632). happensAt(walking(mwt1gt_5),632). orientation(mwt1gt_6,150,632). coords(mwt1gt_6,78,71,632).  visible(mwt1gt_6,632). happensAt(walking(mwt1gt_6),632).
orientation(mwt1gt_5,150,633). coords(mwt1gt_5,82,68,633).  visible(mwt1gt_5,633). happensAt(walking(mwt1gt_5),633). orientation(mwt1gt_6,150,633). coords(mwt1gt_6,78,73,633).  visible(mwt1gt_6,633). happensAt(walking(mwt1gt_6),633).
orientation(mwt1gt_5,150,634). coords(mwt1gt_5,82,68,634).  visible(mwt1gt_5,634). happensAt(walking(mwt1gt_5),634). orientation(mwt1gt_6,150,634). coords(mwt1gt_6,78,73,634).  visible(mwt1gt_6,634). happensAt(walking(mwt1gt_6),634).
orientation(mwt1gt_5,150,635). coords(mwt1gt_5,82,71,635).  visible(mwt1gt_5,635). happensAt(walking(mwt1gt_5),635). orientation(mwt1gt_6,150,635). coords(mwt1gt_6,78,73,635).  visible(mwt1gt_6,635). happensAt(walking(mwt1gt_6),635).
orientation(mwt1gt_5,150,636). coords(mwt1gt_5,82,71,636).  visible(mwt1gt_5,636). happensAt(walking(mwt1gt_5),636). orientation(mwt1gt_6,150,636). coords(mwt1gt_6,78,73,636).  visible(mwt1gt_6,636). happensAt(walking(mwt1gt_6),636).
orientation(mwt1gt_5,150,637). coords(mwt1gt_5,82,71,637).  visible(mwt1gt_5,637). happensAt(walking(mwt1gt_5),637). orientation(mwt1gt_6,150,637). coords(mwt1gt_6,78,73,637).  visible(mwt1gt_6,637). happensAt(walking(mwt1gt_6),637).
orientation(mwt1gt_5,150,638). coords(mwt1gt_5,82,71,638).  visible(mwt1gt_5,638). happensAt(walking(mwt1gt_5),638). orientation(mwt1gt_6,150,638). coords(mwt1gt_6,78,73,638).  visible(mwt1gt_6,638). happensAt(walking(mwt1gt_6),638).
orientation(mwt1gt_5,150,639). coords(mwt1gt_5,82,71,639).  visible(mwt1gt_5,639). happensAt(walking(mwt1gt_5),639). orientation(mwt1gt_6,150,639). coords(mwt1gt_6,78,73,639).  visible(mwt1gt_6,639). happensAt(walking(mwt1gt_6),639).
orientation(mwt1gt_5,150,640). coords(mwt1gt_5,82,71,640).  visible(mwt1gt_5,640). happensAt(walking(mwt1gt_5),640). orientation(mwt1gt_6,150,640). coords(mwt1gt_6,78,73,640).  visible(mwt1gt_6,640). happensAt(walking(mwt1gt_6),640).
orientation(mwt1gt_5,150,641). coords(mwt1gt_5,82,71,641).  visible(mwt1gt_5,641). happensAt(walking(mwt1gt_5),641). orientation(mwt1gt_6,150,641). coords(mwt1gt_6,78,73,641).  visible(mwt1gt_6,641). happensAt(walking(mwt1gt_6),641).
orientation(mwt1gt_5,150,642). coords(mwt1gt_5,82,71,642).  visible(mwt1gt_5,642). happensAt(walking(mwt1gt_5),642). orientation(mwt1gt_6,150,642). coords(mwt1gt_6,78,73,642).  visible(mwt1gt_6,642). happensAt(walking(mwt1gt_6),642).
orientation(mwt1gt_5,150,643). coords(mwt1gt_5,82,71,643).  visible(mwt1gt_5,643). happensAt(walking(mwt1gt_5),643). orientation(mwt1gt_6,150,643). coords(mwt1gt_6,74,78,643).  visible(mwt1gt_6,643). happensAt(walking(mwt1gt_6),643).
orientation(mwt1gt_5,150,644). coords(mwt1gt_5,82,71,644).  visible(mwt1gt_5,644). happensAt(walking(mwt1gt_5),644). orientation(mwt1gt_6,150,644). coords(mwt1gt_6,74,78,644).  visible(mwt1gt_6,644). happensAt(walking(mwt1gt_6),644).
orientation(mwt1gt_5,150,645). coords(mwt1gt_5,82,71,645).  visible(mwt1gt_5,645). happensAt(walking(mwt1gt_5),645). orientation(mwt1gt_6,150,645). coords(mwt1gt_6,74,78,645).  visible(mwt1gt_6,645). happensAt(walking(mwt1gt_6),645).
orientation(mwt1gt_5,150,646). coords(mwt1gt_5,82,71,646).  visible(mwt1gt_5,646). happensAt(walking(mwt1gt_5),646). orientation(mwt1gt_6,150,646). coords(mwt1gt_6,74,78,646).  visible(mwt1gt_6,646). happensAt(walking(mwt1gt_6),646).
orientation(mwt1gt_5,150,647). coords(mwt1gt_5,82,71,647).  visible(mwt1gt_5,647). happensAt(walking(mwt1gt_5),647). orientation(mwt1gt_6,150,647). coords(mwt1gt_6,74,78,647).  visible(mwt1gt_6,647). happensAt(walking(mwt1gt_6),647).
orientation(mwt1gt_5,150,648). coords(mwt1gt_5,82,75,648).  visible(mwt1gt_5,648). happensAt(walking(mwt1gt_5),648). orientation(mwt1gt_6,150,648). coords(mwt1gt_6,74,82,648).  visible(mwt1gt_6,648). happensAt(walking(mwt1gt_6),648).
orientation(mwt1gt_5,150,649). coords(mwt1gt_5,82,75,649).  visible(mwt1gt_5,649). happensAt(walking(mwt1gt_5),649). orientation(mwt1gt_6,150,649). coords(mwt1gt_6,74,82,649).  visible(mwt1gt_6,649). happensAt(walking(mwt1gt_6),649).
#show holdsAt/2.