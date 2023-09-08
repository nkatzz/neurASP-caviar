
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

	orientation(wk2gt_1,125,624). coords(wk2gt_1,115,53,624).  visible(wk2gt_1,624). happensAt(inactive(wk2gt_1),624). orientation(wk2gt_2,142,624). coords(wk2gt_2,99,63,624).  visible(wk2gt_2,624). happensAt(inactive(wk2gt_2),624).
orientation(wk2gt_1,125,625). coords(wk2gt_1,115,53,625).  visible(wk2gt_1,625). happensAt(inactive(wk2gt_1),625). orientation(wk2gt_2,142,625). coords(wk2gt_2,99,63,625).  visible(wk2gt_2,625). happensAt(inactive(wk2gt_2),625).
orientation(wk2gt_1,125,626). coords(wk2gt_1,115,53,626).  visible(wk2gt_1,626). happensAt(inactive(wk2gt_1),626). orientation(wk2gt_2,142,626). coords(wk2gt_2,99,63,626).  visible(wk2gt_2,626). happensAt(inactive(wk2gt_2),626).
orientation(wk2gt_1,125,627). coords(wk2gt_1,115,53,627).  visible(wk2gt_1,627). happensAt(inactive(wk2gt_1),627). orientation(wk2gt_2,142,627). coords(wk2gt_2,99,63,627).  visible(wk2gt_2,627). happensAt(inactive(wk2gt_2),627).
orientation(wk2gt_1,125,628). coords(wk2gt_1,115,53,628).  visible(wk2gt_1,628). happensAt(inactive(wk2gt_1),628). orientation(wk2gt_2,142,628). coords(wk2gt_2,99,63,628).  visible(wk2gt_2,628). happensAt(inactive(wk2gt_2),628).
orientation(wk2gt_1,125,629). coords(wk2gt_1,115,53,629).  visible(wk2gt_1,629). happensAt(inactive(wk2gt_1),629). orientation(wk2gt_2,142,629). coords(wk2gt_2,99,63,629).  visible(wk2gt_2,629). happensAt(inactive(wk2gt_2),629).
orientation(wk2gt_1,125,630). coords(wk2gt_1,115,53,630).  visible(wk2gt_1,630). happensAt(inactive(wk2gt_1),630). orientation(wk2gt_2,142,630). coords(wk2gt_2,99,63,630).  visible(wk2gt_2,630). happensAt(inactive(wk2gt_2),630).
orientation(wk2gt_1,125,631). coords(wk2gt_1,115,53,631).  visible(wk2gt_1,631). happensAt(inactive(wk2gt_1),631). orientation(wk2gt_2,142,631). coords(wk2gt_2,99,63,631).  visible(wk2gt_2,631). happensAt(inactive(wk2gt_2),631).
orientation(wk2gt_1,125,632). coords(wk2gt_1,115,53,632).  visible(wk2gt_1,632). happensAt(inactive(wk2gt_1),632). orientation(wk2gt_2,142,632). coords(wk2gt_2,99,63,632).  visible(wk2gt_2,632). happensAt(inactive(wk2gt_2),632).
orientation(wk2gt_1,125,633). coords(wk2gt_1,115,53,633).  visible(wk2gt_1,633). happensAt(inactive(wk2gt_1),633). orientation(wk2gt_2,142,633). coords(wk2gt_2,99,63,633).  visible(wk2gt_2,633). happensAt(inactive(wk2gt_2),633).
orientation(wk2gt_1,125,634). coords(wk2gt_1,115,53,634).  visible(wk2gt_1,634). happensAt(inactive(wk2gt_1),634). orientation(wk2gt_2,142,634). coords(wk2gt_2,99,63,634).  visible(wk2gt_2,634). happensAt(inactive(wk2gt_2),634).
orientation(wk2gt_1,125,635). coords(wk2gt_1,115,53,635).  visible(wk2gt_1,635). happensAt(inactive(wk2gt_1),635). orientation(wk2gt_2,142,635). coords(wk2gt_2,99,63,635).  visible(wk2gt_2,635). happensAt(inactive(wk2gt_2),635).
orientation(wk2gt_1,125,636). coords(wk2gt_1,115,53,636).  visible(wk2gt_1,636). happensAt(inactive(wk2gt_1),636). orientation(wk2gt_2,142,636). coords(wk2gt_2,99,63,636).  visible(wk2gt_2,636). happensAt(inactive(wk2gt_2),636).
orientation(wk2gt_1,125,637). coords(wk2gt_1,115,53,637).  visible(wk2gt_1,637). happensAt(inactive(wk2gt_1),637). orientation(wk2gt_2,142,637). coords(wk2gt_2,99,63,637).  visible(wk2gt_2,637). happensAt(inactive(wk2gt_2),637).
orientation(wk2gt_1,125,638). coords(wk2gt_1,115,53,638).  visible(wk2gt_1,638). happensAt(inactive(wk2gt_1),638). orientation(wk2gt_2,142,638). coords(wk2gt_2,99,63,638).  visible(wk2gt_2,638). happensAt(inactive(wk2gt_2),638).
orientation(wk2gt_1,125,639). coords(wk2gt_1,115,53,639).  visible(wk2gt_1,639). happensAt(inactive(wk2gt_1),639). orientation(wk2gt_2,142,639). coords(wk2gt_2,99,63,639).  visible(wk2gt_2,639). happensAt(inactive(wk2gt_2),639).
orientation(wk2gt_1,125,640). coords(wk2gt_1,115,53,640).  visible(wk2gt_1,640). happensAt(inactive(wk2gt_1),640). orientation(wk2gt_2,142,640). coords(wk2gt_2,99,63,640).  visible(wk2gt_2,640). happensAt(inactive(wk2gt_2),640).
orientation(wk2gt_1,125,641). coords(wk2gt_1,115,53,641).  visible(wk2gt_1,641). happensAt(inactive(wk2gt_1),641). orientation(wk2gt_2,142,641). coords(wk2gt_2,99,63,641).  visible(wk2gt_2,641). happensAt(inactive(wk2gt_2),641).
orientation(wk2gt_1,125,642). coords(wk2gt_1,114,53,642).  visible(wk2gt_1,642). happensAt(inactive(wk2gt_1),642). orientation(wk2gt_2,142,642). coords(wk2gt_2,99,63,642).  visible(wk2gt_2,642). happensAt(inactive(wk2gt_2),642).
orientation(wk2gt_1,125,643). coords(wk2gt_1,114,53,643).  visible(wk2gt_1,643). happensAt(inactive(wk2gt_1),643). orientation(wk2gt_2,142,643). coords(wk2gt_2,99,63,643).  visible(wk2gt_2,643). happensAt(inactive(wk2gt_2),643).
orientation(wk2gt_1,125,644). coords(wk2gt_1,114,53,644).  visible(wk2gt_1,644). happensAt(inactive(wk2gt_1),644). orientation(wk2gt_2,142,644). coords(wk2gt_2,99,63,644).  visible(wk2gt_2,644). happensAt(inactive(wk2gt_2),644).
orientation(wk2gt_1,125,645). coords(wk2gt_1,114,53,645).  visible(wk2gt_1,645). happensAt(inactive(wk2gt_1),645). orientation(wk2gt_2,142,645). coords(wk2gt_2,99,63,645).  visible(wk2gt_2,645). happensAt(inactive(wk2gt_2),645).
orientation(wk2gt_1,125,646). coords(wk2gt_1,114,53,646).  visible(wk2gt_1,646). happensAt(inactive(wk2gt_1),646). orientation(wk2gt_2,142,646). coords(wk2gt_2,99,63,646).  visible(wk2gt_2,646). happensAt(inactive(wk2gt_2),646).
orientation(wk2gt_1,125,647). coords(wk2gt_1,114,53,647).  visible(wk2gt_1,647). happensAt(inactive(wk2gt_1),647). orientation(wk2gt_2,142,647). coords(wk2gt_2,99,63,647).  visible(wk2gt_2,647). happensAt(inactive(wk2gt_2),647).
orientation(wk2gt_1,125,648). coords(wk2gt_1,114,53,648).  visible(wk2gt_1,648). happensAt(inactive(wk2gt_1),648). orientation(wk2gt_2,142,648). coords(wk2gt_2,99,63,648).  visible(wk2gt_2,648). happensAt(inactive(wk2gt_2),648).
#show holdsAt/2.