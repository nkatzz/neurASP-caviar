

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
orientation(mwt2gt_1,153,685). coords(mwt2gt_1,26,131,685).  visible(mwt2gt_1,685). orientation(mwt2gt_2,152,685). coords(mwt2gt_2,24,138,685).  visible(mwt2gt_2,685).
orientation(mwt2gt_1,153,686). coords(mwt2gt_1,26,130,686).  visible(mwt2gt_1,686). orientation(mwt2gt_2,152,686). coords(mwt2gt_2,24,138,686).  visible(mwt2gt_2,686).
orientation(mwt2gt_1,153,687). coords(mwt2gt_1,26,130,687).  visible(mwt2gt_1,687). orientation(mwt2gt_2,152,687). coords(mwt2gt_2,24,138,687).  visible(mwt2gt_2,687).
orientation(mwt2gt_1,153,688). coords(mwt2gt_1,26,130,688).  visible(mwt2gt_1,688). orientation(mwt2gt_2,152,688). coords(mwt2gt_2,24,138,688).  visible(mwt2gt_2,688).
orientation(mwt2gt_1,153,689). coords(mwt2gt_1,26,130,689).  visible(mwt2gt_1,689). orientation(mwt2gt_2,152,689). coords(mwt2gt_2,24,138,689).  visible(mwt2gt_2,689).
orientation(mwt2gt_1,153,690). coords(mwt2gt_1,26,130,690).  visible(mwt2gt_1,690). orientation(mwt2gt_2,152,690). coords(mwt2gt_2,24,138,690).  visible(mwt2gt_2,690).
orientation(mwt2gt_1,153,691). coords(mwt2gt_1,26,130,691).  visible(mwt2gt_1,691). orientation(mwt2gt_2,152,691). coords(mwt2gt_2,24,138,691).  visible(mwt2gt_2,691).
orientation(mwt2gt_1,153,692). coords(mwt2gt_1,26,130,692).  visible(mwt2gt_1,692). orientation(mwt2gt_2,152,692). coords(mwt2gt_2,24,138,692).  visible(mwt2gt_2,692).
orientation(mwt2gt_1,153,693). coords(mwt2gt_1,26,129,693).  visible(mwt2gt_1,693). orientation(mwt2gt_2,152,693). coords(mwt2gt_2,24,138,693).  visible(mwt2gt_2,693).
orientation(mwt2gt_1,153,694). coords(mwt2gt_1,26,129,694).  visible(mwt2gt_1,694). orientation(mwt2gt_2,152,694). coords(mwt2gt_2,24,138,694).  visible(mwt2gt_2,694).
orientation(mwt2gt_1,153,695). coords(mwt2gt_1,26,129,695).  visible(mwt2gt_1,695). orientation(mwt2gt_2,152,695). coords(mwt2gt_2,24,138,695).  visible(mwt2gt_2,695).
orientation(mwt2gt_1,153,696). coords(mwt2gt_1,26,129,696).  visible(mwt2gt_1,696). orientation(mwt2gt_2,152,696). coords(mwt2gt_2,24,138,696).  visible(mwt2gt_2,696).
orientation(mwt2gt_1,153,697). coords(mwt2gt_1,26,129,697).  visible(mwt2gt_1,697). orientation(mwt2gt_2,152,697). coords(mwt2gt_2,24,138,697).  visible(mwt2gt_2,697).
orientation(mwt2gt_1,153,698). coords(mwt2gt_1,26,129,698).  visible(mwt2gt_1,698). orientation(mwt2gt_2,152,698). coords(mwt2gt_2,24,138,698).  visible(mwt2gt_2,698).
orientation(mwt2gt_1,153,699). coords(mwt2gt_1,26,129,699).  visible(mwt2gt_1,699). orientation(mwt2gt_2,152,699). coords(mwt2gt_2,24,138,699).  visible(mwt2gt_2,699).
orientation(mwt2gt_1,153,700). coords(mwt2gt_1,26,129,700).  visible(mwt2gt_1,700). orientation(mwt2gt_2,152,700). coords(mwt2gt_2,24,138,700).  visible(mwt2gt_2,700).
orientation(mwt2gt_1,153,701). coords(mwt2gt_1,26,129,701).  visible(mwt2gt_1,701). orientation(mwt2gt_2,152,701). coords(mwt2gt_2,24,138,701).  visible(mwt2gt_2,701).
orientation(mwt2gt_1,153,702). coords(mwt2gt_1,26,129,702).  visible(mwt2gt_1,702). orientation(mwt2gt_2,152,702). coords(mwt2gt_2,24,138,702).  visible(mwt2gt_2,702).
orientation(mwt2gt_1,153,703). coords(mwt2gt_1,26,129,703).  visible(mwt2gt_1,703). orientation(mwt2gt_2,152,703). coords(mwt2gt_2,24,138,703).  visible(mwt2gt_2,703).
orientation(mwt2gt_1,153,704). coords(mwt2gt_1,26,129,704).  visible(mwt2gt_1,704). orientation(mwt2gt_2,152,704). coords(mwt2gt_2,24,138,704).  visible(mwt2gt_2,704).
orientation(mwt2gt_1,153,705). coords(mwt2gt_1,26,129,705).  visible(mwt2gt_1,705). orientation(mwt2gt_2,152,705). coords(mwt2gt_2,24,138,705).  visible(mwt2gt_2,705).
orientation(mwt2gt_1,153,706). coords(mwt2gt_1,26,129,706).  visible(mwt2gt_1,706). orientation(mwt2gt_2,152,706). coords(mwt2gt_2,24,138,706).  visible(mwt2gt_2,706).
orientation(mwt2gt_1,153,707). coords(mwt2gt_1,26,129,707).  visible(mwt2gt_1,707). orientation(mwt2gt_2,152,707). coords(mwt2gt_2,24,138,707).  visible(mwt2gt_2,707).
orientation(mwt2gt_1,153,708). coords(mwt2gt_1,26,129,708).  visible(mwt2gt_1,708). orientation(mwt2gt_2,152,708). coords(mwt2gt_2,24,138,708).  visible(mwt2gt_2,708).
orientation(mwt2gt_1,153,709). coords(mwt2gt_1,26,129,709).  visible(mwt2gt_1,709). orientation(mwt2gt_2,152,709). coords(mwt2gt_2,24,138,709).  visible(mwt2gt_2,709).
orientation(mwt2gt_1,153,710). coords(mwt2gt_1,26,129,710).  visible(mwt2gt_1,710). orientation(mwt2gt_2,152,710). coords(mwt2gt_2,24,138,710).  visible(mwt2gt_2,710).
happensAt(disappear(mwt2gt_1),709). happensAt(disappear(mwt2gt_2),709).
1{simpleEvent(T,mwt2gt_1,active); simpleEvent(T,mwt2gt_1,inactive); simpleEvent(T,mwt2gt_1,walking); simpleEvent(T,mwt2gt_1,running)}1 :- T=685..709.
1{simpleEvent(T,mwt2gt_2,active); simpleEvent(T,mwt2gt_2,inactive); simpleEvent(T,mwt2gt_2,walking); simpleEvent(T,mwt2gt_2,running)}1 :- T=685..709.
:- holdsAt(meeting(mwt2gt_1,mwt2gt_2),685). :- holdsAt(meeting(mwt2gt_2,mwt2gt_1),685).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),686). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),686).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),687). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),687).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),688). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),688).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),689). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),689).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),690). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),690).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),691). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),691).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),692). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),692).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),693). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),693).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),694). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),694).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),695). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),695).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),696). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),696).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),697). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),697).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),698). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),698).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),699). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),699).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),700). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),700).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),701). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),701).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),702). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),702).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),703). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),703).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),704). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),704).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),705). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),705).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),706). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),706).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),707). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),707).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),708). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),708).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),709). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),709).
:- holdsAt(meeting(mwt2gt_1,mwt2gt_2),710). :- holdsAt(meeting(mwt2gt_2,mwt2gt_1),710).
:- simpleEvent(T,P,walking), T>685 , T<709.
:- simpleEvent(T,P,running), T>685 , T<709.

#show simpleEvent/3.