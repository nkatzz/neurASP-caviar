

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
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,34,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,34,X2).


happensAt(active(X),T) :- simpleEvent(T,X,active).
happensAt(inactive(X),T) :- simpleEvent(T,X,inactive).
happensAt(walking(X),T) :- simpleEvent(T,X,walking).
happensAt(running(X),T) :- simpleEvent(T,X,running). 


holdsByInertia(moving(X,Y),T+1) :- holdsAt(moving(X,Y),T+1), not initiatedAt(moving(X,Y),T).
holdsByInertia(meeting(X,Y),T+1) :- holdsAt(meeting(X,Y),T+1), not initiatedAt(meeting(X,Y),T).

:- holdsByInertia(moving(X,Y),T+1), simpleEvent(T,X,E1), simpleEvent(T+1,X,E2), E1 != E2. 
:- holdsByInertia(meeting(X,Y),T+1), simpleEvent(T,X,E1), simpleEvent(T+1,X,E2), E1 != E2.

	person(wk2gt_1;wk2gt_2).
orientation(wk2gt_1,125,624). coords(wk2gt_1,115,53,624).  visible(wk2gt_1,624). orientation(wk2gt_2,142,624). coords(wk2gt_2,99,63,624).  visible(wk2gt_2,624).
orientation(wk2gt_1,125,625). coords(wk2gt_1,115,53,625).  visible(wk2gt_1,625). orientation(wk2gt_2,142,625). coords(wk2gt_2,99,63,625).  visible(wk2gt_2,625).
orientation(wk2gt_1,125,626). coords(wk2gt_1,115,53,626).  visible(wk2gt_1,626). orientation(wk2gt_2,142,626). coords(wk2gt_2,99,63,626).  visible(wk2gt_2,626).
orientation(wk2gt_1,125,627). coords(wk2gt_1,115,53,627).  visible(wk2gt_1,627). orientation(wk2gt_2,142,627). coords(wk2gt_2,99,63,627).  visible(wk2gt_2,627).
orientation(wk2gt_1,125,628). coords(wk2gt_1,115,53,628).  visible(wk2gt_1,628). orientation(wk2gt_2,142,628). coords(wk2gt_2,99,63,628).  visible(wk2gt_2,628).
orientation(wk2gt_1,125,629). coords(wk2gt_1,115,53,629).  visible(wk2gt_1,629). orientation(wk2gt_2,142,629). coords(wk2gt_2,99,63,629).  visible(wk2gt_2,629).
orientation(wk2gt_1,125,630). coords(wk2gt_1,115,53,630).  visible(wk2gt_1,630). orientation(wk2gt_2,142,630). coords(wk2gt_2,99,63,630).  visible(wk2gt_2,630).
orientation(wk2gt_1,125,631). coords(wk2gt_1,115,53,631).  visible(wk2gt_1,631). orientation(wk2gt_2,142,631). coords(wk2gt_2,99,63,631).  visible(wk2gt_2,631).
orientation(wk2gt_1,125,632). coords(wk2gt_1,115,53,632).  visible(wk2gt_1,632). orientation(wk2gt_2,142,632). coords(wk2gt_2,99,63,632).  visible(wk2gt_2,632).
orientation(wk2gt_1,125,633). coords(wk2gt_1,115,53,633).  visible(wk2gt_1,633). orientation(wk2gt_2,142,633). coords(wk2gt_2,99,63,633).  visible(wk2gt_2,633).
orientation(wk2gt_1,125,634). coords(wk2gt_1,115,53,634).  visible(wk2gt_1,634). orientation(wk2gt_2,142,634). coords(wk2gt_2,99,63,634).  visible(wk2gt_2,634).
orientation(wk2gt_1,125,635). coords(wk2gt_1,115,53,635).  visible(wk2gt_1,635). orientation(wk2gt_2,142,635). coords(wk2gt_2,99,63,635).  visible(wk2gt_2,635).
orientation(wk2gt_1,125,636). coords(wk2gt_1,115,53,636).  visible(wk2gt_1,636). orientation(wk2gt_2,142,636). coords(wk2gt_2,99,63,636).  visible(wk2gt_2,636).
orientation(wk2gt_1,125,637). coords(wk2gt_1,115,53,637).  visible(wk2gt_1,637). orientation(wk2gt_2,142,637). coords(wk2gt_2,99,63,637).  visible(wk2gt_2,637).
orientation(wk2gt_1,125,638). coords(wk2gt_1,115,53,638).  visible(wk2gt_1,638). orientation(wk2gt_2,142,638). coords(wk2gt_2,99,63,638).  visible(wk2gt_2,638).
orientation(wk2gt_1,125,639). coords(wk2gt_1,115,53,639).  visible(wk2gt_1,639). orientation(wk2gt_2,142,639). coords(wk2gt_2,99,63,639).  visible(wk2gt_2,639).
orientation(wk2gt_1,125,640). coords(wk2gt_1,115,53,640).  visible(wk2gt_1,640). orientation(wk2gt_2,142,640). coords(wk2gt_2,99,63,640).  visible(wk2gt_2,640).
orientation(wk2gt_1,125,641). coords(wk2gt_1,115,53,641).  visible(wk2gt_1,641). orientation(wk2gt_2,142,641). coords(wk2gt_2,99,63,641).  visible(wk2gt_2,641).
orientation(wk2gt_1,125,642). coords(wk2gt_1,114,53,642).  visible(wk2gt_1,642). orientation(wk2gt_2,142,642). coords(wk2gt_2,99,63,642).  visible(wk2gt_2,642).
orientation(wk2gt_1,125,643). coords(wk2gt_1,114,53,643).  visible(wk2gt_1,643). orientation(wk2gt_2,142,643). coords(wk2gt_2,99,63,643).  visible(wk2gt_2,643).
orientation(wk2gt_1,125,644). coords(wk2gt_1,114,53,644).  visible(wk2gt_1,644). orientation(wk2gt_2,142,644). coords(wk2gt_2,99,63,644).  visible(wk2gt_2,644).
orientation(wk2gt_1,125,645). coords(wk2gt_1,114,53,645).  visible(wk2gt_1,645). orientation(wk2gt_2,142,645). coords(wk2gt_2,99,63,645).  visible(wk2gt_2,645).
orientation(wk2gt_1,125,646). coords(wk2gt_1,114,53,646).  visible(wk2gt_1,646). orientation(wk2gt_2,142,646). coords(wk2gt_2,99,63,646).  visible(wk2gt_2,646).
orientation(wk2gt_1,125,647). coords(wk2gt_1,114,53,647).  visible(wk2gt_1,647). orientation(wk2gt_2,142,647). coords(wk2gt_2,99,63,647).  visible(wk2gt_2,647).
orientation(wk2gt_1,125,648). coords(wk2gt_1,114,53,648).  visible(wk2gt_1,648). orientation(wk2gt_2,142,648). coords(wk2gt_2,99,63,648).  visible(wk2gt_2,648).
orientation(wk2gt_1,125,649). coords(wk2gt_1,114,53,649).  visible(wk2gt_1,649). orientation(wk2gt_2,142,649). coords(wk2gt_2,99,63,649).  visible(wk2gt_2,649).
happensAt(disappear(wk2gt_1),648). happensAt(disappear(wk2gt_2),648).
1{simpleEvent(T,wk2gt_1,active); simpleEvent(T,wk2gt_1,inactive); simpleEvent(T,wk2gt_1,walking); simpleEvent(T,wk2gt_1,running)}1 :- T=624..648.
1{simpleEvent(T,wk2gt_2,active); simpleEvent(T,wk2gt_2,inactive); simpleEvent(T,wk2gt_2,walking); simpleEvent(T,wk2gt_2,running)}1 :- T=624..648.
:- holdsAt(meeting(wk2gt_1,wk2gt_2),624). :- holdsAt(meeting(wk2gt_2,wk2gt_1),624).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),625). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),625).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),626). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),626).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),627). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),627).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),628). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),628).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),629). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),629).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),630). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),630).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),631). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),631).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),632). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),632).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),633). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),633).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),634). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),634).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),635). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),635).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),636). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),636).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),637). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),637).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),638). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),638).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),639). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),639).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),640). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),640).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),641). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),641).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),642). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),642).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),643). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),643).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),644). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),644).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),645). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),645).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),646). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),646).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),647). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),647).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),648). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),648).
:- holdsAt(meeting(wk2gt_1,wk2gt_2),649). :- holdsAt(meeting(wk2gt_2,wk2gt_1),649).
:- simpleEvent(T,P,walking), T>624 , T<648.
:- simpleEvent(T,P,running), T>624 , T<648.

#show simpleEvent/3.