

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
orientation(mws1gt_5,161,228). coords(mws1gt_5,45,124,228).  visible(mws1gt_5,228). orientation(mws1gt_6,146,228). coords(mws1gt_6,52,101,228).  visible(mws1gt_6,228).
orientation(mws1gt_5,161,229). coords(mws1gt_5,45,126,229).  visible(mws1gt_5,229). orientation(mws1gt_6,146,229). coords(mws1gt_6,52,102,229).  visible(mws1gt_6,229).
orientation(mws1gt_5,161,230). coords(mws1gt_5,45,127,230).  visible(mws1gt_5,230). orientation(mws1gt_6,146,230). coords(mws1gt_6,48,102,230).  visible(mws1gt_6,230).
orientation(mws1gt_5,161,231). coords(mws1gt_5,45,128,231).  visible(mws1gt_5,231). orientation(mws1gt_6,146,231). coords(mws1gt_6,48,103,231).  visible(mws1gt_6,231).
orientation(mws1gt_5,161,232). coords(mws1gt_5,45,128,232).  visible(mws1gt_5,232). orientation(mws1gt_6,146,232). coords(mws1gt_6,48,103,232).  visible(mws1gt_6,232).
orientation(mws1gt_5,161,233). coords(mws1gt_5,45,129,233).  visible(mws1gt_5,233). orientation(mws1gt_6,146,233). coords(mws1gt_6,48,103,233).  visible(mws1gt_6,233).
orientation(mws1gt_5,161,234). coords(mws1gt_5,45,129,234).  visible(mws1gt_5,234). orientation(mws1gt_6,146,234). coords(mws1gt_6,48,105,234).  visible(mws1gt_6,234).
orientation(mws1gt_5,161,235). coords(mws1gt_5,45,130,235).  visible(mws1gt_5,235). orientation(mws1gt_6,146,235). coords(mws1gt_6,48,106,235).  visible(mws1gt_6,235).
orientation(mws1gt_5,161,236). coords(mws1gt_5,44,131,236).  visible(mws1gt_5,236). orientation(mws1gt_6,146,236). coords(mws1gt_6,48,106,236).  visible(mws1gt_6,236).
orientation(mws1gt_5,161,237). coords(mws1gt_5,43,132,237).  visible(mws1gt_5,237). orientation(mws1gt_6,146,237). coords(mws1gt_6,44,107,237).  visible(mws1gt_6,237).
orientation(mws1gt_5,161,238). coords(mws1gt_5,43,133,238).  visible(mws1gt_5,238). orientation(mws1gt_6,146,238). coords(mws1gt_6,44,108,238).  visible(mws1gt_6,238).
orientation(mws1gt_5,161,239). coords(mws1gt_5,43,134,239).  visible(mws1gt_5,239). orientation(mws1gt_6,146,239). coords(mws1gt_6,44,109,239).  visible(mws1gt_6,239).
orientation(mws1gt_5,161,240). coords(mws1gt_5,43,134,240).  visible(mws1gt_5,240). orientation(mws1gt_6,146,240). coords(mws1gt_6,44,110,240).  visible(mws1gt_6,240).
orientation(mws1gt_5,161,241). coords(mws1gt_5,43,134,241).  visible(mws1gt_5,241). orientation(mws1gt_6,146,241). coords(mws1gt_6,44,111,241).  visible(mws1gt_6,241).
orientation(mws1gt_5,161,242). coords(mws1gt_5,42,135,242).  visible(mws1gt_5,242). orientation(mws1gt_6,146,242). coords(mws1gt_6,43,112,242).  visible(mws1gt_6,242).
orientation(mws1gt_5,161,243). coords(mws1gt_5,42,135,243).  visible(mws1gt_5,243). orientation(mws1gt_6,146,243). coords(mws1gt_6,43,112,243).  visible(mws1gt_6,243).
orientation(mws1gt_5,161,244). coords(mws1gt_5,42,135,244).  visible(mws1gt_5,244). orientation(mws1gt_6,146,244). coords(mws1gt_6,43,112,244).  visible(mws1gt_6,244).
orientation(mws1gt_5,161,245). coords(mws1gt_5,42,135,245).  visible(mws1gt_5,245). orientation(mws1gt_6,146,245). coords(mws1gt_6,43,112,245).  visible(mws1gt_6,245).
orientation(mws1gt_5,161,246). coords(mws1gt_5,41,135,246).  visible(mws1gt_5,246). orientation(mws1gt_6,146,246). coords(mws1gt_6,43,112,246).  visible(mws1gt_6,246).
orientation(mws1gt_5,161,247). coords(mws1gt_5,41,137,247).  visible(mws1gt_5,247). orientation(mws1gt_6,146,247). coords(mws1gt_6,43,112,247).  visible(mws1gt_6,247).
orientation(mws1gt_5,161,248). coords(mws1gt_5,41,138,248).  visible(mws1gt_5,248). orientation(mws1gt_6,146,248). coords(mws1gt_6,43,112,248).  visible(mws1gt_6,248).
orientation(mws1gt_5,161,249). coords(mws1gt_5,41,138,249).  visible(mws1gt_5,249). orientation(mws1gt_6,146,249). coords(mws1gt_6,43,114,249).  visible(mws1gt_6,249).
orientation(mws1gt_5,161,250). coords(mws1gt_5,41,139,250).  visible(mws1gt_5,250). orientation(mws1gt_6,146,250). coords(mws1gt_6,43,115,250).  visible(mws1gt_6,250).
orientation(mws1gt_5,161,251). coords(mws1gt_5,41,140,251).  visible(mws1gt_5,251). orientation(mws1gt_6,146,251). coords(mws1gt_6,42,116,251).  visible(mws1gt_6,251).
orientation(mws1gt_5,161,252). coords(mws1gt_5,41,141,252).  visible(mws1gt_5,252). orientation(mws1gt_6,146,252). coords(mws1gt_6,42,117,252).  visible(mws1gt_6,252).
orientation(mws1gt_5,161,253). coords(mws1gt_5,41,141,253).  visible(mws1gt_5,253). orientation(mws1gt_6,146,253). coords(mws1gt_6,42,117,253).  visible(mws1gt_6,253).
happensAt(disappear(mws1gt_5),252). happensAt(disappear(mws1gt_6),252).
1{simpleEvent(T,mws1gt_5,active); simpleEvent(T,mws1gt_5,inactive); simpleEvent(T,mws1gt_5,walking); simpleEvent(T,mws1gt_5,running)}1 :- T=228..252.
1{simpleEvent(T,mws1gt_6,active); simpleEvent(T,mws1gt_6,inactive); simpleEvent(T,mws1gt_6,walking); simpleEvent(T,mws1gt_6,running)}1 :- T=228..252.
:- holdsAt(moving(mws1gt_5,mws1gt_6),228). :- holdsAt(moving(mws1gt_6,mws1gt_5),228).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),229). :- not holdsAt(moving(mws1gt_6,mws1gt_5),229).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),230). :- not holdsAt(moving(mws1gt_6,mws1gt_5),230).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),231). :- not holdsAt(moving(mws1gt_6,mws1gt_5),231).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),232). :- not holdsAt(moving(mws1gt_6,mws1gt_5),232).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),233). :- not holdsAt(moving(mws1gt_6,mws1gt_5),233).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),234). :- not holdsAt(moving(mws1gt_6,mws1gt_5),234).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),235). :- not holdsAt(moving(mws1gt_6,mws1gt_5),235).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),236). :- not holdsAt(moving(mws1gt_6,mws1gt_5),236).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),237). :- not holdsAt(moving(mws1gt_6,mws1gt_5),237).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),238). :- not holdsAt(moving(mws1gt_6,mws1gt_5),238).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),239). :- not holdsAt(moving(mws1gt_6,mws1gt_5),239).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),240). :- not holdsAt(moving(mws1gt_6,mws1gt_5),240).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),241). :- not holdsAt(moving(mws1gt_6,mws1gt_5),241).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),242). :- not holdsAt(moving(mws1gt_6,mws1gt_5),242).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),243). :- not holdsAt(moving(mws1gt_6,mws1gt_5),243).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),244). :- not holdsAt(moving(mws1gt_6,mws1gt_5),244).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),245). :- not holdsAt(moving(mws1gt_6,mws1gt_5),245).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),246). :- not holdsAt(moving(mws1gt_6,mws1gt_5),246).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),247). :- not holdsAt(moving(mws1gt_6,mws1gt_5),247).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),248). :- not holdsAt(moving(mws1gt_6,mws1gt_5),248).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),249). :- not holdsAt(moving(mws1gt_6,mws1gt_5),249).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),250). :- not holdsAt(moving(mws1gt_6,mws1gt_5),250).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),251). :- not holdsAt(moving(mws1gt_6,mws1gt_5),251).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),252). :- not holdsAt(moving(mws1gt_6,mws1gt_5),252).
:- holdsAt(moving(mws1gt_5,mws1gt_6),253). :- holdsAt(moving(mws1gt_6,mws1gt_5),253).
:- simpleEvent(T,P,inactive), T>228 , T<252.
:- simpleEvent(T,P,running), T>228 , T<252.

#show simpleEvent/3.