

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

	person(mws1gt_5;mws1gt_6).
orientation(mws1gt_5,161,252). coords(mws1gt_5,41,141,252).  visible(mws1gt_5,252). orientation(mws1gt_6,146,252). coords(mws1gt_6,42,117,252).  visible(mws1gt_6,252).
orientation(mws1gt_5,161,253). coords(mws1gt_5,41,142,253).  visible(mws1gt_5,253). orientation(mws1gt_6,146,253). coords(mws1gt_6,42,119,253).  visible(mws1gt_6,253).
orientation(mws1gt_5,161,254). coords(mws1gt_5,41,143,254).  visible(mws1gt_5,254). orientation(mws1gt_6,146,254). coords(mws1gt_6,42,119,254).  visible(mws1gt_6,254).
orientation(mws1gt_5,161,255). coords(mws1gt_5,41,143,255).  visible(mws1gt_5,255). orientation(mws1gt_6,146,255). coords(mws1gt_6,43,120,255).  visible(mws1gt_6,255).
orientation(mws1gt_5,161,256). coords(mws1gt_5,41,143,256).  visible(mws1gt_5,256). orientation(mws1gt_6,146,256). coords(mws1gt_6,43,120,256).  visible(mws1gt_6,256).
orientation(mws1gt_5,161,257). coords(mws1gt_5,41,143,257).  visible(mws1gt_5,257). orientation(mws1gt_6,146,257). coords(mws1gt_6,43,121,257).  visible(mws1gt_6,257).
orientation(mws1gt_5,161,258). coords(mws1gt_5,41,143,258).  visible(mws1gt_5,258). orientation(mws1gt_6,146,258). coords(mws1gt_6,43,121,258).  visible(mws1gt_6,258).
orientation(mws1gt_5,161,259). coords(mws1gt_5,41,144,259).  visible(mws1gt_5,259). orientation(mws1gt_6,146,259). coords(mws1gt_6,43,124,259).  visible(mws1gt_6,259).
orientation(mws1gt_5,161,260). coords(mws1gt_5,41,147,260).  visible(mws1gt_5,260). orientation(mws1gt_6,146,260). coords(mws1gt_6,43,125,260).  visible(mws1gt_6,260).
orientation(mws1gt_5,161,261). coords(mws1gt_5,41,148,261).  visible(mws1gt_5,261). orientation(mws1gt_6,146,261). coords(mws1gt_6,43,126,261).  visible(mws1gt_6,261).
orientation(mws1gt_5,161,262). coords(mws1gt_5,41,150,262).  visible(mws1gt_5,262). orientation(mws1gt_6,146,262). coords(mws1gt_6,43,128,262).  visible(mws1gt_6,262).
orientation(mws1gt_5,161,263). coords(mws1gt_5,41,150,263).  visible(mws1gt_5,263). orientation(mws1gt_6,146,263). coords(mws1gt_6,42,129,263).  visible(mws1gt_6,263).
orientation(mws1gt_5,161,264). coords(mws1gt_5,41,151,264).  visible(mws1gt_5,264). orientation(mws1gt_6,146,264). coords(mws1gt_6,42,129,264).  visible(mws1gt_6,264).
orientation(mws1gt_5,161,265). coords(mws1gt_5,41,153,265).  visible(mws1gt_5,265). orientation(mws1gt_6,146,265). coords(mws1gt_6,42,129,265).  visible(mws1gt_6,265).
orientation(mws1gt_5,161,266). coords(mws1gt_5,41,153,266).  visible(mws1gt_5,266). orientation(mws1gt_6,146,266). coords(mws1gt_6,42,130,266).  visible(mws1gt_6,266).
orientation(mws1gt_5,161,267). coords(mws1gt_5,41,153,267).  visible(mws1gt_5,267). orientation(mws1gt_6,146,267). coords(mws1gt_6,42,130,267).  visible(mws1gt_6,267).
orientation(mws1gt_5,161,268). coords(mws1gt_5,41,153,268).  visible(mws1gt_5,268). orientation(mws1gt_6,146,268). coords(mws1gt_6,42,130,268).  visible(mws1gt_6,268).
orientation(mws1gt_5,161,269). coords(mws1gt_5,41,153,269).  visible(mws1gt_5,269). orientation(mws1gt_6,146,269). coords(mws1gt_6,42,131,269).  visible(mws1gt_6,269).
orientation(mws1gt_5,161,270). coords(mws1gt_5,40,153,270).  visible(mws1gt_5,270). orientation(mws1gt_6,146,270). coords(mws1gt_6,42,132,270).  visible(mws1gt_6,270).
orientation(mws1gt_5,161,271). coords(mws1gt_5,40,154,271).  visible(mws1gt_5,271). orientation(mws1gt_6,146,271). coords(mws1gt_6,41,133,271).  visible(mws1gt_6,271).
orientation(mws1gt_5,161,272). coords(mws1gt_5,40,154,272).  visible(mws1gt_5,272). orientation(mws1gt_6,146,272). coords(mws1gt_6,41,133,272).  visible(mws1gt_6,272).
orientation(mws1gt_5,161,273). coords(mws1gt_5,40,155,273).  visible(mws1gt_5,273). orientation(mws1gt_6,146,273). coords(mws1gt_6,40,134,273).  visible(mws1gt_6,273).
orientation(mws1gt_5,161,274). coords(mws1gt_5,40,156,274).  visible(mws1gt_5,274). orientation(mws1gt_6,146,274). coords(mws1gt_6,40,134,274).  visible(mws1gt_6,274).
orientation(mws1gt_5,161,275). coords(mws1gt_5,40,157,275).  visible(mws1gt_5,275). orientation(mws1gt_6,146,275). coords(mws1gt_6,40,135,275).  visible(mws1gt_6,275).
orientation(mws1gt_5,161,276). coords(mws1gt_5,39,158,276).  visible(mws1gt_5,276). orientation(mws1gt_6,146,276). coords(mws1gt_6,40,136,276).  visible(mws1gt_6,276).
orientation(mws1gt_5,161,277). coords(mws1gt_5,39,158,277).  visible(mws1gt_5,277). orientation(mws1gt_6,146,277). coords(mws1gt_6,40,136,277).  visible(mws1gt_6,277).
happensAt(disappear(mws1gt_5),276). happensAt(disappear(mws1gt_6),276).
1{simpleEvent(T,mws1gt_5,active); simpleEvent(T,mws1gt_5,inactive); simpleEvent(T,mws1gt_5,walking); simpleEvent(T,mws1gt_5,running)}1 :- T=252..276.
1{simpleEvent(T,mws1gt_6,active); simpleEvent(T,mws1gt_6,inactive); simpleEvent(T,mws1gt_6,walking); simpleEvent(T,mws1gt_6,running)}1 :- T=252..276.
:- holdsAt(moving(mws1gt_5,mws1gt_6),252). :- holdsAt(moving(mws1gt_6,mws1gt_5),252).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),253). :- not holdsAt(moving(mws1gt_6,mws1gt_5),253).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),254). :- not holdsAt(moving(mws1gt_6,mws1gt_5),254).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),255). :- not holdsAt(moving(mws1gt_6,mws1gt_5),255).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),256). :- not holdsAt(moving(mws1gt_6,mws1gt_5),256).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),257). :- not holdsAt(moving(mws1gt_6,mws1gt_5),257).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),258). :- not holdsAt(moving(mws1gt_6,mws1gt_5),258).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),259). :- not holdsAt(moving(mws1gt_6,mws1gt_5),259).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),260). :- not holdsAt(moving(mws1gt_6,mws1gt_5),260).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),261). :- not holdsAt(moving(mws1gt_6,mws1gt_5),261).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),262). :- not holdsAt(moving(mws1gt_6,mws1gt_5),262).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),263). :- not holdsAt(moving(mws1gt_6,mws1gt_5),263).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),264). :- not holdsAt(moving(mws1gt_6,mws1gt_5),264).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),265). :- not holdsAt(moving(mws1gt_6,mws1gt_5),265).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),266). :- not holdsAt(moving(mws1gt_6,mws1gt_5),266).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),267). :- not holdsAt(moving(mws1gt_6,mws1gt_5),267).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),268). :- not holdsAt(moving(mws1gt_6,mws1gt_5),268).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),269). :- not holdsAt(moving(mws1gt_6,mws1gt_5),269).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),270). :- not holdsAt(moving(mws1gt_6,mws1gt_5),270).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),271). :- not holdsAt(moving(mws1gt_6,mws1gt_5),271).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),272). :- not holdsAt(moving(mws1gt_6,mws1gt_5),272).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),273). :- not holdsAt(moving(mws1gt_6,mws1gt_5),273).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),274). :- not holdsAt(moving(mws1gt_6,mws1gt_5),274).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),275). :- not holdsAt(moving(mws1gt_6,mws1gt_5),275).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),276). :- not holdsAt(moving(mws1gt_6,mws1gt_5),276).
:- holdsAt(moving(mws1gt_5,mws1gt_6),277). :- holdsAt(moving(mws1gt_6,mws1gt_5),277).
:- simpleEvent(T,P,inactive), T>252 , T<276.
:- simpleEvent(T,P,running), T>252 , T<276.

#show simpleEvent/3.