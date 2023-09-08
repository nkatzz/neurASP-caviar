

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

	person(ms3ggt_0;ms3ggt_1).
orientation(ms3ggt_0,154,254). coords(ms3ggt_0,58,153,254).  visible(ms3ggt_0,254). orientation(ms3ggt_1,167,254). coords(ms3ggt_1,55,165,254).  visible(ms3ggt_1,254).
orientation(ms3ggt_0,154,255). coords(ms3ggt_0,59,153,255).  visible(ms3ggt_0,255). orientation(ms3ggt_1,167,255). coords(ms3ggt_1,55,166,255).  visible(ms3ggt_1,255).
orientation(ms3ggt_0,154,256). coords(ms3ggt_0,59,153,256).  visible(ms3ggt_0,256). orientation(ms3ggt_1,167,256). coords(ms3ggt_1,55,166,256).  visible(ms3ggt_1,256).
orientation(ms3ggt_0,154,257). coords(ms3ggt_0,59,153,257).  visible(ms3ggt_0,257). orientation(ms3ggt_1,155,257). coords(ms3ggt_1,55,166,257).  visible(ms3ggt_1,257).
orientation(ms3ggt_0,154,258). coords(ms3ggt_0,60,153,258).  visible(ms3ggt_0,258). orientation(ms3ggt_1,155,258). coords(ms3ggt_1,55,166,258).  visible(ms3ggt_1,258).
orientation(ms3ggt_0,154,259). coords(ms3ggt_0,60,153,259).  visible(ms3ggt_0,259). orientation(ms3ggt_1,155,259). coords(ms3ggt_1,56,166,259).  visible(ms3ggt_1,259).
orientation(ms3ggt_0,154,260). coords(ms3ggt_0,62,152,260).  visible(ms3ggt_0,260). orientation(ms3ggt_1,155,260). coords(ms3ggt_1,56,166,260).  visible(ms3ggt_1,260).
orientation(ms3ggt_0,154,261). coords(ms3ggt_0,62,152,261).  visible(ms3ggt_0,261). orientation(ms3ggt_1,155,261). coords(ms3ggt_1,57,166,261).  visible(ms3ggt_1,261).
orientation(ms3ggt_0,154,262). coords(ms3ggt_0,62,152,262).  visible(ms3ggt_0,262). orientation(ms3ggt_1,155,262). coords(ms3ggt_1,59,166,262).  visible(ms3ggt_1,262).
orientation(ms3ggt_0,157,263). coords(ms3ggt_0,63,151,263).  visible(ms3ggt_0,263). orientation(ms3ggt_1,158,263). coords(ms3ggt_1,60,166,263).  visible(ms3ggt_1,263).
orientation(ms3ggt_0,157,264). coords(ms3ggt_0,63,151,264).  visible(ms3ggt_0,264). orientation(ms3ggt_1,158,264). coords(ms3ggt_1,61,166,264).  visible(ms3ggt_1,264).
orientation(ms3ggt_0,157,265). coords(ms3ggt_0,63,151,265).  visible(ms3ggt_0,265). orientation(ms3ggt_1,158,265). coords(ms3ggt_1,61,165,265).  visible(ms3ggt_1,265).
orientation(ms3ggt_0,157,266). coords(ms3ggt_0,63,151,266).  visible(ms3ggt_0,266). orientation(ms3ggt_1,158,266). coords(ms3ggt_1,62,165,266).  visible(ms3ggt_1,266).
orientation(ms3ggt_0,157,267). coords(ms3ggt_0,63,151,267).  visible(ms3ggt_0,267). orientation(ms3ggt_1,158,267). coords(ms3ggt_1,62,165,267).  visible(ms3ggt_1,267).
orientation(ms3ggt_0,157,268). coords(ms3ggt_0,63,151,268).  visible(ms3ggt_0,268). orientation(ms3ggt_1,158,268). coords(ms3ggt_1,62,166,268).  visible(ms3ggt_1,268).
orientation(ms3ggt_0,157,269). coords(ms3ggt_0,63,150,269).  visible(ms3ggt_0,269). orientation(ms3ggt_1,158,269). coords(ms3ggt_1,62,166,269).  visible(ms3ggt_1,269).
orientation(ms3ggt_0,157,270). coords(ms3ggt_0,63,150,270).  visible(ms3ggt_0,270). orientation(ms3ggt_1,158,270). coords(ms3ggt_1,62,166,270).  visible(ms3ggt_1,270).
orientation(ms3ggt_0,157,271). coords(ms3ggt_0,63,150,271).  visible(ms3ggt_0,271). orientation(ms3ggt_1,158,271). coords(ms3ggt_1,62,166,271).  visible(ms3ggt_1,271).
orientation(ms3ggt_0,157,272). coords(ms3ggt_0,65,150,272).  visible(ms3ggt_0,272). orientation(ms3ggt_1,158,272). coords(ms3ggt_1,62,166,272).  visible(ms3ggt_1,272).
orientation(ms3ggt_0,157,273). coords(ms3ggt_0,66,150,273).  visible(ms3ggt_0,273). orientation(ms3ggt_1,158,273). coords(ms3ggt_1,63,165,273).  visible(ms3ggt_1,273).
orientation(ms3ggt_0,161,274). coords(ms3ggt_0,67,150,274).  visible(ms3ggt_0,274). orientation(ms3ggt_1,158,274). coords(ms3ggt_1,63,164,274).  visible(ms3ggt_1,274).
orientation(ms3ggt_0,161,275). coords(ms3ggt_0,67,150,275).  visible(ms3ggt_0,275). orientation(ms3ggt_1,158,275). coords(ms3ggt_1,63,164,275).  visible(ms3ggt_1,275).
orientation(ms3ggt_0,161,276). coords(ms3ggt_0,68,150,276).  visible(ms3ggt_0,276). orientation(ms3ggt_1,158,276). coords(ms3ggt_1,63,164,276).  visible(ms3ggt_1,276).
orientation(ms3ggt_0,161,277). coords(ms3ggt_0,68,150,277).  visible(ms3ggt_0,277). orientation(ms3ggt_1,158,277). coords(ms3ggt_1,66,164,277).  visible(ms3ggt_1,277).
orientation(ms3ggt_0,161,278). coords(ms3ggt_0,68,149,278).  visible(ms3ggt_0,278). orientation(ms3ggt_1,158,278). coords(ms3ggt_1,66,164,278).  visible(ms3ggt_1,278).
orientation(ms3ggt_0,161,279). coords(ms3ggt_0,68,149,279).  visible(ms3ggt_0,279). orientation(ms3ggt_1,158,279). coords(ms3ggt_1,66,164,279).  visible(ms3ggt_1,279).
happensAt(disappear(ms3ggt_0),278). happensAt(disappear(ms3ggt_1),278).
1{simpleEvent(T,ms3ggt_0,active); simpleEvent(T,ms3ggt_0,inactive); simpleEvent(T,ms3ggt_0,walking); simpleEvent(T,ms3ggt_0,running)}1 :- T=254..278.
1{simpleEvent(T,ms3ggt_1,active); simpleEvent(T,ms3ggt_1,inactive); simpleEvent(T,ms3ggt_1,walking); simpleEvent(T,ms3ggt_1,running)}1 :- T=254..278.
:- holdsAt(moving(ms3ggt_0,ms3ggt_1),254). :- holdsAt(moving(ms3ggt_1,ms3ggt_0),254).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),255). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),255).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),256). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),256).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),257). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),257).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),258). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),258).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),259). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),259).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),260). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),260).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),261). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),261).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),262). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),262).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),263). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),263).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),264). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),264).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),265). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),265).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),266). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),266).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),267). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),267).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),268). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),268).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),269). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),269).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),270). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),270).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),271). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),271).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),272). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),272).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),273). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),273).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),274). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),274).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),275). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),275).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),276). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),276).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),277). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),277).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),278). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),278).
:- holdsAt(moving(ms3ggt_0,ms3ggt_1),279). :- holdsAt(moving(ms3ggt_1,ms3ggt_0),279).
:- simpleEvent(T,P,inactive), T>254 , T<278.
:- simpleEvent(T,P,running), T>254 , T<278.

#show simpleEvent/3.