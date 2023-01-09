
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

	orientation(fcgt_1,163,232). coords(fcgt_1,189,190,232).  visible(fcgt_1,232). happensAt(walking(fcgt_1),232). orientation(fcgt_2,157,232). coords(fcgt_2,53,172,232).  visible(fcgt_2,232). happensAt(walking(fcgt_2),232).
orientation(fcgt_1,156,233). coords(fcgt_1,188,188,233).  visible(fcgt_1,233). happensAt(walking(fcgt_1),233). orientation(fcgt_2,157,233). coords(fcgt_2,53,171,233).  visible(fcgt_2,233). happensAt(walking(fcgt_2),233).
orientation(fcgt_1,156,234). coords(fcgt_1,186,188,234).  visible(fcgt_1,234). happensAt(walking(fcgt_1),234). orientation(fcgt_2,157,234). coords(fcgt_2,53,170,234).  visible(fcgt_2,234). happensAt(walking(fcgt_2),234).
orientation(fcgt_1,149,235). coords(fcgt_1,186,186,235).  visible(fcgt_1,235). happensAt(walking(fcgt_1),235). orientation(fcgt_2,157,235). coords(fcgt_2,53,169,235).  visible(fcgt_2,235). happensAt(walking(fcgt_2),235).
orientation(fcgt_1,149,236). coords(fcgt_1,184,183,236).  visible(fcgt_1,236). happensAt(walking(fcgt_1),236). orientation(fcgt_2,157,236). coords(fcgt_2,53,169,236).  visible(fcgt_2,236). happensAt(walking(fcgt_2),236).
orientation(fcgt_1,149,237). coords(fcgt_1,181,181,237).  visible(fcgt_1,237). happensAt(walking(fcgt_1),237). orientation(fcgt_2,157,237). coords(fcgt_2,55,168,237).  visible(fcgt_2,237). happensAt(walking(fcgt_2),237).
orientation(fcgt_1,149,238). coords(fcgt_1,177,178,238).  visible(fcgt_1,238). happensAt(walking(fcgt_1),238). orientation(fcgt_2,157,238). coords(fcgt_2,56,167,238).  visible(fcgt_2,238). happensAt(walking(fcgt_2),238).
orientation(fcgt_1,149,239). coords(fcgt_1,175,175,239).  visible(fcgt_1,239). happensAt(walking(fcgt_1),239). orientation(fcgt_2,157,239). coords(fcgt_2,56,167,239).  visible(fcgt_2,239). happensAt(walking(fcgt_2),239).
orientation(fcgt_1,149,240). coords(fcgt_1,173,171,240).  visible(fcgt_1,240). happensAt(walking(fcgt_1),240). orientation(fcgt_2,157,240). coords(fcgt_2,56,166,240).  visible(fcgt_2,240). happensAt(walking(fcgt_2),240).
orientation(fcgt_1,149,241). coords(fcgt_1,173,170,241).  visible(fcgt_1,241). happensAt(walking(fcgt_1),241). orientation(fcgt_2,157,241). coords(fcgt_2,57,165,241).  visible(fcgt_2,241). happensAt(walking(fcgt_2),241).
orientation(fcgt_1,149,242). coords(fcgt_1,172,165,242).  visible(fcgt_1,242). happensAt(walking(fcgt_1),242). orientation(fcgt_2,157,242). coords(fcgt_2,57,165,242).  visible(fcgt_2,242). happensAt(walking(fcgt_2),242).
orientation(fcgt_1,149,243). coords(fcgt_1,171,163,243).  visible(fcgt_1,243). happensAt(walking(fcgt_1),243). orientation(fcgt_2,157,243). coords(fcgt_2,57,163,243).  visible(fcgt_2,243). happensAt(walking(fcgt_2),243).
orientation(fcgt_1,154,244). coords(fcgt_1,171,162,244).  visible(fcgt_1,244). happensAt(walking(fcgt_1),244). orientation(fcgt_2,157,244). coords(fcgt_2,58,163,244).  visible(fcgt_2,244). happensAt(walking(fcgt_2),244).
orientation(fcgt_1,154,245). coords(fcgt_1,170,161,245).  visible(fcgt_1,245). happensAt(walking(fcgt_1),245). orientation(fcgt_2,157,245). coords(fcgt_2,58,162,245).  visible(fcgt_2,245). happensAt(walking(fcgt_2),245).
orientation(fcgt_1,154,246). coords(fcgt_1,169,160,246).  visible(fcgt_1,246). happensAt(walking(fcgt_1),246). orientation(fcgt_2,157,246). coords(fcgt_2,58,161,246).  visible(fcgt_2,246). happensAt(walking(fcgt_2),246).
orientation(fcgt_1,154,247). coords(fcgt_1,167,159,247).  visible(fcgt_1,247). happensAt(walking(fcgt_1),247). orientation(fcgt_2,157,247). coords(fcgt_2,58,161,247).  visible(fcgt_2,247). happensAt(walking(fcgt_2),247).
orientation(fcgt_1,154,248). coords(fcgt_1,165,157,248).  visible(fcgt_1,248). happensAt(walking(fcgt_1),248). orientation(fcgt_2,157,248). coords(fcgt_2,59,161,248).  visible(fcgt_2,248). happensAt(walking(fcgt_2),248).
orientation(fcgt_1,157,249). coords(fcgt_1,164,155,249).  visible(fcgt_1,249). happensAt(walking(fcgt_1),249). orientation(fcgt_2,157,249). coords(fcgt_2,61,160,249).  visible(fcgt_2,249). happensAt(walking(fcgt_2),249).
orientation(fcgt_1,157,250). coords(fcgt_1,159,154,250).  visible(fcgt_1,250). happensAt(walking(fcgt_1),250). orientation(fcgt_2,157,250). coords(fcgt_2,61,159,250).  visible(fcgt_2,250). happensAt(walking(fcgt_2),250).
orientation(fcgt_1,157,251). coords(fcgt_1,157,152,251).  visible(fcgt_1,251). happensAt(walking(fcgt_1),251). orientation(fcgt_2,157,251). coords(fcgt_2,62,159,251).  visible(fcgt_2,251). happensAt(walking(fcgt_2),251).
orientation(fcgt_1,157,252). coords(fcgt_1,154,153,252).  visible(fcgt_1,252). happensAt(walking(fcgt_1),252). orientation(fcgt_2,157,252). coords(fcgt_2,63,158,252).  visible(fcgt_2,252). happensAt(walking(fcgt_2),252).
orientation(fcgt_1,157,253). coords(fcgt_1,154,152,253).  visible(fcgt_1,253). happensAt(walking(fcgt_1),253). orientation(fcgt_2,157,253). coords(fcgt_2,63,158,253).  visible(fcgt_2,253). happensAt(walking(fcgt_2),253).
orientation(fcgt_1,157,254). coords(fcgt_1,153,151,254).  visible(fcgt_1,254). happensAt(walking(fcgt_1),254). orientation(fcgt_2,157,254). coords(fcgt_2,63,157,254).  visible(fcgt_2,254). happensAt(walking(fcgt_2),254).
orientation(fcgt_1,157,255). coords(fcgt_1,152,151,255).  visible(fcgt_1,255). happensAt(walking(fcgt_1),255). orientation(fcgt_2,157,255). coords(fcgt_2,64,156,255).  visible(fcgt_2,255). happensAt(walking(fcgt_2),255).
orientation(fcgt_1,157,256). coords(fcgt_1,152,150,256).  visible(fcgt_1,256). happensAt(walking(fcgt_1),256). orientation(fcgt_2,157,256). coords(fcgt_2,64,155,256).  visible(fcgt_2,256). happensAt(walking(fcgt_2),256).
#show holdsAt/2.