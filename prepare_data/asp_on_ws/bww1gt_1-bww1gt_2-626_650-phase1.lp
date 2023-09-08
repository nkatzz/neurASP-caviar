
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

	orientation(bww1gt_1,173,626). coords(bww1gt_1,101,255,626).  visible(bww1gt_1,626). happensAt(walking(bww1gt_1),626). orientation(bww1gt_2,169,626). coords(bww1gt_2,152,208,626).  visible(bww1gt_2,626). happensAt(walking(bww1gt_2),626).
orientation(bww1gt_1,173,627). coords(bww1gt_1,101,255,627).  visible(bww1gt_1,627). happensAt(walking(bww1gt_1),627). orientation(bww1gt_2,169,627). coords(bww1gt_2,155,209,627).  visible(bww1gt_2,627). happensAt(walking(bww1gt_2),627).
orientation(bww1gt_1,173,628). coords(bww1gt_1,101,255,628).  visible(bww1gt_1,628). happensAt(walking(bww1gt_1),628). orientation(bww1gt_2,169,628). coords(bww1gt_2,157,209,628).  visible(bww1gt_2,628). happensAt(walking(bww1gt_2),628).
orientation(bww1gt_1,173,629). coords(bww1gt_1,102,256,629).  visible(bww1gt_1,629). happensAt(walking(bww1gt_1),629). orientation(bww1gt_2,169,629). coords(bww1gt_2,160,211,629).  visible(bww1gt_2,629). happensAt(walking(bww1gt_2),629).
orientation(bww1gt_1,173,630). coords(bww1gt_1,102,256,630).  visible(bww1gt_1,630). happensAt(walking(bww1gt_1),630). orientation(bww1gt_2,169,630). coords(bww1gt_2,164,211,630).  visible(bww1gt_2,630). happensAt(walking(bww1gt_2),630).
orientation(bww1gt_1,173,631). coords(bww1gt_1,102,256,631).  visible(bww1gt_1,631). happensAt(walking(bww1gt_1),631). orientation(bww1gt_2,169,631). coords(bww1gt_2,164,211,631).  visible(bww1gt_2,631). happensAt(walking(bww1gt_2),631).
orientation(bww1gt_1,173,632). coords(bww1gt_1,102,256,632).  visible(bww1gt_1,632). happensAt(walking(bww1gt_1),632). orientation(bww1gt_2,169,632). coords(bww1gt_2,165,211,632).  visible(bww1gt_2,632). happensAt(walking(bww1gt_2),632).
orientation(bww1gt_1,173,633). coords(bww1gt_1,102,256,633).  visible(bww1gt_1,633). happensAt(walking(bww1gt_1),633). orientation(bww1gt_2,169,633). coords(bww1gt_2,167,212,633).  visible(bww1gt_2,633). happensAt(walking(bww1gt_2),633).
orientation(bww1gt_1,173,634). coords(bww1gt_1,102,256,634).  visible(bww1gt_1,634). happensAt(walking(bww1gt_1),634). orientation(bww1gt_2,169,634). coords(bww1gt_2,167,212,634).  visible(bww1gt_2,634). happensAt(walking(bww1gt_2),634).
orientation(bww1gt_1,173,635). coords(bww1gt_1,102,257,635).  visible(bww1gt_1,635). happensAt(walking(bww1gt_1),635). orientation(bww1gt_2,169,635). coords(bww1gt_2,168,213,635).  visible(bww1gt_2,635). happensAt(walking(bww1gt_2),635).
orientation(bww1gt_1,173,636). coords(bww1gt_1,102,257,636).  visible(bww1gt_1,636). happensAt(walking(bww1gt_1),636). orientation(bww1gt_2,169,636). coords(bww1gt_2,169,215,636).  visible(bww1gt_2,636). happensAt(walking(bww1gt_2),636).
orientation(bww1gt_1,173,637). coords(bww1gt_1,102,257,637).  visible(bww1gt_1,637). happensAt(walking(bww1gt_1),637). orientation(bww1gt_2,169,637). coords(bww1gt_2,169,215,637).  visible(bww1gt_2,637). happensAt(walking(bww1gt_2),637).
orientation(bww1gt_1,173,638). coords(bww1gt_1,102,257,638).  visible(bww1gt_1,638). happensAt(walking(bww1gt_1),638). orientation(bww1gt_2,169,638). coords(bww1gt_2,170,216,638).  visible(bww1gt_2,638). happensAt(walking(bww1gt_2),638).
orientation(bww1gt_1,173,639). coords(bww1gt_1,102,257,639).  visible(bww1gt_1,639). happensAt(walking(bww1gt_1),639). orientation(bww1gt_2,169,639). coords(bww1gt_2,174,217,639).  visible(bww1gt_2,639). happensAt(walking(bww1gt_2),639).
orientation(bww1gt_1,173,640). coords(bww1gt_1,102,257,640).  visible(bww1gt_1,640). happensAt(walking(bww1gt_1),640). orientation(bww1gt_2,169,640). coords(bww1gt_2,178,217,640).  visible(bww1gt_2,640). happensAt(walking(bww1gt_2),640).
orientation(bww1gt_1,173,641). coords(bww1gt_1,102,257,641).  visible(bww1gt_1,641). happensAt(walking(bww1gt_1),641). orientation(bww1gt_2,169,641). coords(bww1gt_2,180,217,641).  visible(bww1gt_2,641). happensAt(walking(bww1gt_2),641).
orientation(bww1gt_1,173,642). coords(bww1gt_1,102,257,642).  visible(bww1gt_1,642). happensAt(walking(bww1gt_1),642). orientation(bww1gt_2,169,642). coords(bww1gt_2,183,216,642).  visible(bww1gt_2,642). happensAt(walking(bww1gt_2),642).
orientation(bww1gt_1,173,643). coords(bww1gt_1,102,257,643).  visible(bww1gt_1,643). happensAt(walking(bww1gt_1),643). orientation(bww1gt_2,169,643). coords(bww1gt_2,185,218,643).  visible(bww1gt_2,643). happensAt(walking(bww1gt_2),643).
orientation(bww1gt_1,173,644). coords(bww1gt_1,102,257,644).  visible(bww1gt_1,644). happensAt(walking(bww1gt_1),644). orientation(bww1gt_2,169,644). coords(bww1gt_2,187,218,644).  visible(bww1gt_2,644). happensAt(walking(bww1gt_2),644).
orientation(bww1gt_1,173,645). coords(bww1gt_1,102,257,645).  visible(bww1gt_1,645). happensAt(walking(bww1gt_1),645). orientation(bww1gt_2,169,645). coords(bww1gt_2,188,219,645).  visible(bww1gt_2,645). happensAt(walking(bww1gt_2),645).
orientation(bww1gt_1,173,646). coords(bww1gt_1,102,257,646).  visible(bww1gt_1,646). happensAt(walking(bww1gt_1),646). orientation(bww1gt_2,169,646). coords(bww1gt_2,189,220,646).  visible(bww1gt_2,646). happensAt(walking(bww1gt_2),646).
orientation(bww1gt_1,173,647). coords(bww1gt_1,102,257,647).  visible(bww1gt_1,647). happensAt(walking(bww1gt_1),647). orientation(bww1gt_2,169,647). coords(bww1gt_2,190,221,647).  visible(bww1gt_2,647). happensAt(walking(bww1gt_2),647).
orientation(bww1gt_1,173,648). coords(bww1gt_1,102,257,648).  visible(bww1gt_1,648). happensAt(walking(bww1gt_1),648). orientation(bww1gt_2,169,648). coords(bww1gt_2,190,223,648).  visible(bww1gt_2,648). happensAt(walking(bww1gt_2),648).
orientation(bww1gt_1,173,649). coords(bww1gt_1,102,257,649).  visible(bww1gt_1,649). happensAt(walking(bww1gt_1),649). orientation(bww1gt_2,169,649). coords(bww1gt_2,191,224,649).  visible(bww1gt_2,649). happensAt(walking(bww1gt_2),649).
orientation(bww1gt_1,173,650). coords(bww1gt_1,102,257,650).  visible(bww1gt_1,650). happensAt(walking(bww1gt_1),650). orientation(bww1gt_2,169,650). coords(bww1gt_2,192,226,650).  visible(bww1gt_2,650). happensAt(walking(bww1gt_2),650).
#show holdsAt/2.