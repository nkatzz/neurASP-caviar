

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

	person(mwt2gt_1;mwt2gt_2).
orientation(mwt2gt_1,153,625). coords(mwt2gt_1,28,153,625).  visible(mwt2gt_1,625). orientation(mwt2gt_2,152,625). coords(mwt2gt_2,25,154,625).  visible(mwt2gt_2,625).
orientation(mwt2gt_1,153,626). coords(mwt2gt_1,28,152,626).  visible(mwt2gt_1,626). orientation(mwt2gt_2,152,626). coords(mwt2gt_2,25,153,626).  visible(mwt2gt_2,626).
orientation(mwt2gt_1,153,627). coords(mwt2gt_1,28,152,627).  visible(mwt2gt_1,627). orientation(mwt2gt_2,152,627). coords(mwt2gt_2,25,152,627).  visible(mwt2gt_2,627).
orientation(mwt2gt_1,153,628). coords(mwt2gt_1,28,151,628).  visible(mwt2gt_1,628). orientation(mwt2gt_2,152,628). coords(mwt2gt_2,25,151,628).  visible(mwt2gt_2,628).
orientation(mwt2gt_1,153,629). coords(mwt2gt_1,28,151,629).  visible(mwt2gt_1,629). orientation(mwt2gt_2,152,629). coords(mwt2gt_2,25,151,629).  visible(mwt2gt_2,629).
orientation(mwt2gt_1,153,630). coords(mwt2gt_1,28,151,630).  visible(mwt2gt_1,630). orientation(mwt2gt_2,152,630). coords(mwt2gt_2,25,151,630).  visible(mwt2gt_2,630).
orientation(mwt2gt_1,153,631). coords(mwt2gt_1,28,150,631).  visible(mwt2gt_1,631). orientation(mwt2gt_2,152,631). coords(mwt2gt_2,25,151,631).  visible(mwt2gt_2,631).
orientation(mwt2gt_1,153,632). coords(mwt2gt_1,28,150,632).  visible(mwt2gt_1,632). orientation(mwt2gt_2,152,632). coords(mwt2gt_2,25,150,632).  visible(mwt2gt_2,632).
orientation(mwt2gt_1,153,633). coords(mwt2gt_1,28,150,633).  visible(mwt2gt_1,633). orientation(mwt2gt_2,152,633). coords(mwt2gt_2,25,149,633).  visible(mwt2gt_2,633).
orientation(mwt2gt_1,153,634). coords(mwt2gt_1,28,148,634).  visible(mwt2gt_1,634). orientation(mwt2gt_2,152,634). coords(mwt2gt_2,25,148,634).  visible(mwt2gt_2,634).
orientation(mwt2gt_1,153,635). coords(mwt2gt_1,27,148,635).  visible(mwt2gt_1,635). orientation(mwt2gt_2,152,635). coords(mwt2gt_2,25,148,635).  visible(mwt2gt_2,635).
orientation(mwt2gt_1,153,636). coords(mwt2gt_1,27,148,636).  visible(mwt2gt_1,636). orientation(mwt2gt_2,152,636). coords(mwt2gt_2,26,148,636).  visible(mwt2gt_2,636).
orientation(mwt2gt_1,153,637). coords(mwt2gt_1,27,148,637).  visible(mwt2gt_1,637). orientation(mwt2gt_2,152,637). coords(mwt2gt_2,26,148,637).  visible(mwt2gt_2,637).
orientation(mwt2gt_1,153,638). coords(mwt2gt_1,27,146,638).  visible(mwt2gt_1,638). orientation(mwt2gt_2,152,638). coords(mwt2gt_2,26,147,638).  visible(mwt2gt_2,638).
orientation(mwt2gt_1,153,639). coords(mwt2gt_1,27,145,639).  visible(mwt2gt_1,639). orientation(mwt2gt_2,152,639). coords(mwt2gt_2,26,146,639).  visible(mwt2gt_2,639).
orientation(mwt2gt_1,153,640). coords(mwt2gt_1,22,145,640).  visible(mwt2gt_1,640). orientation(mwt2gt_2,152,640). coords(mwt2gt_2,21,146,640).  visible(mwt2gt_2,640).
orientation(mwt2gt_1,153,641). coords(mwt2gt_1,22,143,641).  visible(mwt2gt_1,641). orientation(mwt2gt_2,152,641). coords(mwt2gt_2,21,145,641).  visible(mwt2gt_2,641).
orientation(mwt2gt_1,153,642). coords(mwt2gt_1,22,143,642).  visible(mwt2gt_1,642). orientation(mwt2gt_2,152,642). coords(mwt2gt_2,21,143,642).  visible(mwt2gt_2,642).
orientation(mwt2gt_1,153,643). coords(mwt2gt_1,22,143,643).  visible(mwt2gt_1,643). orientation(mwt2gt_2,152,643). coords(mwt2gt_2,21,143,643).  visible(mwt2gt_2,643).
orientation(mwt2gt_1,153,644). coords(mwt2gt_1,22,142,644).  visible(mwt2gt_1,644). orientation(mwt2gt_2,152,644). coords(mwt2gt_2,21,143,644).  visible(mwt2gt_2,644).
orientation(mwt2gt_1,153,645). coords(mwt2gt_1,22,141,645).  visible(mwt2gt_1,645). orientation(mwt2gt_2,152,645). coords(mwt2gt_2,21,143,645).  visible(mwt2gt_2,645).
orientation(mwt2gt_1,153,646). coords(mwt2gt_1,22,140,646).  visible(mwt2gt_1,646). orientation(mwt2gt_2,152,646). coords(mwt2gt_2,22,143,646).  visible(mwt2gt_2,646).
orientation(mwt2gt_1,153,647). coords(mwt2gt_1,22,140,647).  visible(mwt2gt_1,647). orientation(mwt2gt_2,152,647). coords(mwt2gt_2,22,143,647).  visible(mwt2gt_2,647).
orientation(mwt2gt_1,153,648). coords(mwt2gt_1,22,140,648).  visible(mwt2gt_1,648). orientation(mwt2gt_2,152,648). coords(mwt2gt_2,22,143,648).  visible(mwt2gt_2,648).
orientation(mwt2gt_1,153,649). coords(mwt2gt_1,22,139,649).  visible(mwt2gt_1,649). orientation(mwt2gt_2,152,649). coords(mwt2gt_2,22,141,649).  visible(mwt2gt_2,649).
orientation(mwt2gt_1,153,650). coords(mwt2gt_1,22,139,650).  visible(mwt2gt_1,650). orientation(mwt2gt_2,152,650). coords(mwt2gt_2,22,141,650).  visible(mwt2gt_2,650).
happensAt(disappear(mwt2gt_1),649). happensAt(disappear(mwt2gt_2),649).
1{simpleEvent(T,mwt2gt_1,active); simpleEvent(T,mwt2gt_1,inactive); simpleEvent(T,mwt2gt_1,walking); simpleEvent(T,mwt2gt_1,running)}1 :- T=625..649.
1{simpleEvent(T,mwt2gt_2,active); simpleEvent(T,mwt2gt_2,inactive); simpleEvent(T,mwt2gt_2,walking); simpleEvent(T,mwt2gt_2,running)}1 :- T=625..649.
:- holdsAt(moving(mwt2gt_1,mwt2gt_2),625). :- holdsAt(moving(mwt2gt_2,mwt2gt_1),625).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),626). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),626).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),627). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),627).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),628). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),628).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),629). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),629).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),630). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),630).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),631). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),631).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),632). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),632).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),633). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),633).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),634). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),634).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),635). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),635).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),636). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),636).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),637). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),637).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),638). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),638).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),639). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),639).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),640). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),640).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),641). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),641).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),642). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),642).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),643). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),643).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),644). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),644).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),645). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),645).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),646). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),646).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),647). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),647).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),648). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),648).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),649). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),649).
:- holdsAt(moving(mwt2gt_1,mwt2gt_2),650). :- holdsAt(moving(mwt2gt_2,mwt2gt_1),650).
:- simpleEvent(T,P,inactive), T>625 , T<649.
:- simpleEvent(T,P,running), T>625 , T<649.

#show simpleEvent/3.