

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

	person(mc1gt_0;mc1gt_1).
orientation(mc1gt_0,147,232). coords(mc1gt_0,162,135,232).  visible(mc1gt_0,232). orientation(mc1gt_1,142,232). coords(mc1gt_1,185,135,232).  visible(mc1gt_1,232).
orientation(mc1gt_0,147,233). coords(mc1gt_0,163,136,233).  visible(mc1gt_0,233). orientation(mc1gt_1,142,233). coords(mc1gt_1,186,137,233).  visible(mc1gt_1,233).
orientation(mc1gt_0,147,234). coords(mc1gt_0,164,137,234).  visible(mc1gt_0,234). orientation(mc1gt_1,142,234). coords(mc1gt_1,187,137,234).  visible(mc1gt_1,234).
orientation(mc1gt_0,147,235). coords(mc1gt_0,165,138,235).  visible(mc1gt_0,235). orientation(mc1gt_1,142,235). coords(mc1gt_1,188,138,235).  visible(mc1gt_1,235).
orientation(mc1gt_0,147,236). coords(mc1gt_0,165,138,236).  visible(mc1gt_0,236). orientation(mc1gt_1,142,236). coords(mc1gt_1,189,139,236).  visible(mc1gt_1,236).
orientation(mc1gt_0,147,237). coords(mc1gt_0,166,139,237).  visible(mc1gt_0,237). orientation(mc1gt_1,142,237). coords(mc1gt_1,189,140,237).  visible(mc1gt_1,237).
orientation(mc1gt_0,147,238). coords(mc1gt_0,166,139,238).  visible(mc1gt_0,238). orientation(mc1gt_1,142,238). coords(mc1gt_1,190,141,238).  visible(mc1gt_1,238).
orientation(mc1gt_0,147,239). coords(mc1gt_0,166,140,239).  visible(mc1gt_0,239). orientation(mc1gt_1,142,239). coords(mc1gt_1,190,141,239).  visible(mc1gt_1,239).
orientation(mc1gt_0,147,240). coords(mc1gt_0,166,140,240).  visible(mc1gt_0,240). orientation(mc1gt_1,145,240). coords(mc1gt_1,190,142,240).  visible(mc1gt_1,240).
orientation(mc1gt_0,147,241). coords(mc1gt_0,167,143,241).  visible(mc1gt_0,241). orientation(mc1gt_1,147,241). coords(mc1gt_1,191,142,241).  visible(mc1gt_1,241).
orientation(mc1gt_0,147,242). coords(mc1gt_0,167,146,242).  visible(mc1gt_0,242). orientation(mc1gt_1,146,242). coords(mc1gt_1,192,143,242).  visible(mc1gt_1,242).
orientation(mc1gt_0,147,243). coords(mc1gt_0,167,149,243).  visible(mc1gt_0,243). orientation(mc1gt_1,148,243). coords(mc1gt_1,194,144,243).  visible(mc1gt_1,243).
orientation(mc1gt_0,147,244). coords(mc1gt_0,168,151,244).  visible(mc1gt_0,244). orientation(mc1gt_1,145,244). coords(mc1gt_1,197,145,244).  visible(mc1gt_1,244).
orientation(mc1gt_0,143,245). coords(mc1gt_0,169,152,245).  visible(mc1gt_0,245). orientation(mc1gt_1,142,245). coords(mc1gt_1,198,149,245).  visible(mc1gt_1,245).
orientation(mc1gt_0,143,246). coords(mc1gt_0,170,153,246).  visible(mc1gt_0,246). orientation(mc1gt_1,142,246). coords(mc1gt_1,200,151,246).  visible(mc1gt_1,246).
orientation(mc1gt_0,143,247). coords(mc1gt_0,171,155,247).  visible(mc1gt_0,247). orientation(mc1gt_1,142,247). coords(mc1gt_1,201,154,247).  visible(mc1gt_1,247).
orientation(mc1gt_0,143,248). coords(mc1gt_0,171,156,248).  visible(mc1gt_0,248). orientation(mc1gt_1,142,248). coords(mc1gt_1,202,155,248).  visible(mc1gt_1,248).
orientation(mc1gt_0,143,249). coords(mc1gt_0,171,156,249).  visible(mc1gt_0,249). orientation(mc1gt_1,142,249). coords(mc1gt_1,203,157,249).  visible(mc1gt_1,249).
orientation(mc1gt_0,143,250). coords(mc1gt_0,171,157,250).  visible(mc1gt_0,250). orientation(mc1gt_1,142,250). coords(mc1gt_1,204,158,250).  visible(mc1gt_1,250).
orientation(mc1gt_0,143,251). coords(mc1gt_0,171,157,251).  visible(mc1gt_0,251). orientation(mc1gt_1,142,251). coords(mc1gt_1,205,159,251).  visible(mc1gt_1,251).
orientation(mc1gt_0,144,252). coords(mc1gt_0,172,158,252).  visible(mc1gt_0,252). orientation(mc1gt_1,144,252). coords(mc1gt_1,205,159,252).  visible(mc1gt_1,252).
orientation(mc1gt_0,145,253). coords(mc1gt_0,173,159,253).  visible(mc1gt_0,253). orientation(mc1gt_1,145,253). coords(mc1gt_1,206,160,253).  visible(mc1gt_1,253).
orientation(mc1gt_0,148,254). coords(mc1gt_0,173,159,254).  visible(mc1gt_0,254). orientation(mc1gt_1,146,254). coords(mc1gt_1,206,161,254).  visible(mc1gt_1,254).
orientation(mc1gt_0,149,255). coords(mc1gt_0,175,161,255).  visible(mc1gt_0,255). orientation(mc1gt_1,147,255). coords(mc1gt_1,206,162,255).  visible(mc1gt_1,255).
orientation(mc1gt_0,149,256). coords(mc1gt_0,176,162,256).  visible(mc1gt_0,256). orientation(mc1gt_1,146,256). coords(mc1gt_1,206,163,256).  visible(mc1gt_1,256).
orientation(mc1gt_0,149,257). coords(mc1gt_0,176,162,257).  visible(mc1gt_0,257). orientation(mc1gt_1,146,257). coords(mc1gt_1,206,163,257).  visible(mc1gt_1,257).
happensAt(disappear(mc1gt_0),256). happensAt(disappear(mc1gt_1),256).
1{simpleEvent(T,mc1gt_0,active); simpleEvent(T,mc1gt_0,inactive); simpleEvent(T,mc1gt_0,walking); simpleEvent(T,mc1gt_0,running)}1 :- T=232..256.
1{simpleEvent(T,mc1gt_1,active); simpleEvent(T,mc1gt_1,inactive); simpleEvent(T,mc1gt_1,walking); simpleEvent(T,mc1gt_1,running)}1 :- T=232..256.
:- holdsAt(moving(mc1gt_0,mc1gt_1),232). :- holdsAt(moving(mc1gt_1,mc1gt_0),232).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),233). :- not holdsAt(moving(mc1gt_1,mc1gt_0),233).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),234). :- not holdsAt(moving(mc1gt_1,mc1gt_0),234).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),235). :- not holdsAt(moving(mc1gt_1,mc1gt_0),235).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),236). :- not holdsAt(moving(mc1gt_1,mc1gt_0),236).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),237). :- not holdsAt(moving(mc1gt_1,mc1gt_0),237).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),238). :- not holdsAt(moving(mc1gt_1,mc1gt_0),238).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),239). :- not holdsAt(moving(mc1gt_1,mc1gt_0),239).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),240). :- not holdsAt(moving(mc1gt_1,mc1gt_0),240).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),241). :- not holdsAt(moving(mc1gt_1,mc1gt_0),241).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),242). :- not holdsAt(moving(mc1gt_1,mc1gt_0),242).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),243). :- not holdsAt(moving(mc1gt_1,mc1gt_0),243).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),244). :- not holdsAt(moving(mc1gt_1,mc1gt_0),244).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),245). :- not holdsAt(moving(mc1gt_1,mc1gt_0),245).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),246). :- not holdsAt(moving(mc1gt_1,mc1gt_0),246).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),247). :- not holdsAt(moving(mc1gt_1,mc1gt_0),247).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),248). :- not holdsAt(moving(mc1gt_1,mc1gt_0),248).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),249). :- not holdsAt(moving(mc1gt_1,mc1gt_0),249).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),250). :- not holdsAt(moving(mc1gt_1,mc1gt_0),250).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),251). :- not holdsAt(moving(mc1gt_1,mc1gt_0),251).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),252). :- not holdsAt(moving(mc1gt_1,mc1gt_0),252).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),253). :- not holdsAt(moving(mc1gt_1,mc1gt_0),253).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),254). :- not holdsAt(moving(mc1gt_1,mc1gt_0),254).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),255). :- not holdsAt(moving(mc1gt_1,mc1gt_0),255).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),256). :- not holdsAt(moving(mc1gt_1,mc1gt_0),256).
:- holdsAt(moving(mc1gt_0,mc1gt_1),257). :- holdsAt(moving(mc1gt_1,mc1gt_0),257).
:- simpleEvent(T,P,inactive), T>232 , T<256.
:- simpleEvent(T,P,running), T>232 , T<256.

#show simpleEvent/3.