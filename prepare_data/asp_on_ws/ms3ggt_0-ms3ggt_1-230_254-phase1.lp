
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

	orientation(ms3ggt_0,154,230). coords(ms3ggt_0,58,155,230).  visible(ms3ggt_0,230). happensAt(inactive(ms3ggt_0),230). orientation(ms3ggt_1,163,230). coords(ms3ggt_1,46,164,230).  visible(ms3ggt_1,230). happensAt(walking(ms3ggt_1),230).
orientation(ms3ggt_0,154,231). coords(ms3ggt_0,58,155,231).  visible(ms3ggt_0,231). happensAt(inactive(ms3ggt_0),231). orientation(ms3ggt_1,163,231). coords(ms3ggt_1,46,164,231).  visible(ms3ggt_1,231). happensAt(walking(ms3ggt_1),231).
orientation(ms3ggt_0,154,232). coords(ms3ggt_0,58,155,232).  visible(ms3ggt_0,232). happensAt(inactive(ms3ggt_0),232). orientation(ms3ggt_1,163,232). coords(ms3ggt_1,47,163,232).  visible(ms3ggt_1,232). happensAt(walking(ms3ggt_1),232).
orientation(ms3ggt_0,154,233). coords(ms3ggt_0,58,155,233).  visible(ms3ggt_0,233). happensAt(inactive(ms3ggt_0),233). orientation(ms3ggt_1,163,233). coords(ms3ggt_1,48,164,233).  visible(ms3ggt_1,233). happensAt(walking(ms3ggt_1),233).
orientation(ms3ggt_0,154,234). coords(ms3ggt_0,58,155,234).  visible(ms3ggt_0,234). happensAt(inactive(ms3ggt_0),234). orientation(ms3ggt_1,163,234). coords(ms3ggt_1,48,164,234).  visible(ms3ggt_1,234). happensAt(walking(ms3ggt_1),234).
orientation(ms3ggt_0,154,235). coords(ms3ggt_0,58,155,235).  visible(ms3ggt_0,235). happensAt(inactive(ms3ggt_0),235). orientation(ms3ggt_1,167,235). coords(ms3ggt_1,50,164,235).  visible(ms3ggt_1,235). happensAt(walking(ms3ggt_1),235).
orientation(ms3ggt_0,154,236). coords(ms3ggt_0,58,155,236).  visible(ms3ggt_0,236). happensAt(inactive(ms3ggt_0),236). orientation(ms3ggt_1,167,236). coords(ms3ggt_1,50,164,236).  visible(ms3ggt_1,236). happensAt(walking(ms3ggt_1),236).
orientation(ms3ggt_0,154,237). coords(ms3ggt_0,58,155,237).  visible(ms3ggt_0,237). happensAt(inactive(ms3ggt_0),237). orientation(ms3ggt_1,167,237). coords(ms3ggt_1,50,164,237).  visible(ms3ggt_1,237). happensAt(walking(ms3ggt_1),237).
orientation(ms3ggt_0,154,238). coords(ms3ggt_0,58,155,238).  visible(ms3ggt_0,238). happensAt(inactive(ms3ggt_0),238). orientation(ms3ggt_1,167,238). coords(ms3ggt_1,51,164,238).  visible(ms3ggt_1,238). happensAt(walking(ms3ggt_1),238).
orientation(ms3ggt_0,154,239). coords(ms3ggt_0,58,155,239).  visible(ms3ggt_0,239). happensAt(inactive(ms3ggt_0),239). orientation(ms3ggt_1,167,239). coords(ms3ggt_1,52,164,239).  visible(ms3ggt_1,239). happensAt(walking(ms3ggt_1),239).
orientation(ms3ggt_0,154,240). coords(ms3ggt_0,58,155,240).  visible(ms3ggt_0,240). happensAt(inactive(ms3ggt_0),240). orientation(ms3ggt_1,167,240). coords(ms3ggt_1,52,165,240).  visible(ms3ggt_1,240). happensAt(walking(ms3ggt_1),240).
orientation(ms3ggt_0,154,241). coords(ms3ggt_0,58,155,241).  visible(ms3ggt_0,241). happensAt(inactive(ms3ggt_0),241). orientation(ms3ggt_1,167,241). coords(ms3ggt_1,52,165,241).  visible(ms3ggt_1,241). happensAt(walking(ms3ggt_1),241).
orientation(ms3ggt_0,154,242). coords(ms3ggt_0,58,155,242).  visible(ms3ggt_0,242). happensAt(inactive(ms3ggt_0),242). orientation(ms3ggt_1,167,242). coords(ms3ggt_1,52,165,242).  visible(ms3ggt_1,242). happensAt(walking(ms3ggt_1),242).
orientation(ms3ggt_0,154,243). coords(ms3ggt_0,58,155,243).  visible(ms3ggt_0,243). happensAt(inactive(ms3ggt_0),243). orientation(ms3ggt_1,167,243). coords(ms3ggt_1,52,165,243).  visible(ms3ggt_1,243). happensAt(walking(ms3ggt_1),243).
orientation(ms3ggt_0,154,244). coords(ms3ggt_0,58,155,244).  visible(ms3ggt_0,244). happensAt(inactive(ms3ggt_0),244). orientation(ms3ggt_1,167,244). coords(ms3ggt_1,52,165,244).  visible(ms3ggt_1,244). happensAt(walking(ms3ggt_1),244).
orientation(ms3ggt_0,154,245). coords(ms3ggt_0,58,154,245).  visible(ms3ggt_0,245). happensAt(inactive(ms3ggt_0),245). orientation(ms3ggt_1,167,245). coords(ms3ggt_1,52,165,245).  visible(ms3ggt_1,245). happensAt(walking(ms3ggt_1),245).
orientation(ms3ggt_0,154,246). coords(ms3ggt_0,58,154,246).  visible(ms3ggt_0,246). happensAt(inactive(ms3ggt_0),246). orientation(ms3ggt_1,167,246). coords(ms3ggt_1,52,165,246).  visible(ms3ggt_1,246). happensAt(walking(ms3ggt_1),246).
orientation(ms3ggt_0,154,247). coords(ms3ggt_0,58,154,247).  visible(ms3ggt_0,247). happensAt(inactive(ms3ggt_0),247). orientation(ms3ggt_1,167,247). coords(ms3ggt_1,52,165,247).  visible(ms3ggt_1,247). happensAt(walking(ms3ggt_1),247).
orientation(ms3ggt_0,154,248). coords(ms3ggt_0,58,154,248).  visible(ms3ggt_0,248). happensAt(inactive(ms3ggt_0),248). orientation(ms3ggt_1,167,248). coords(ms3ggt_1,52,165,248).  visible(ms3ggt_1,248). happensAt(walking(ms3ggt_1),248).
orientation(ms3ggt_0,154,249). coords(ms3ggt_0,58,154,249).  visible(ms3ggt_0,249). happensAt(inactive(ms3ggt_0),249). orientation(ms3ggt_1,167,249). coords(ms3ggt_1,53,165,249).  visible(ms3ggt_1,249). happensAt(walking(ms3ggt_1),249).
orientation(ms3ggt_0,154,250). coords(ms3ggt_0,58,154,250).  visible(ms3ggt_0,250). happensAt(inactive(ms3ggt_0),250). orientation(ms3ggt_1,167,250). coords(ms3ggt_1,53,165,250).  visible(ms3ggt_1,250). happensAt(walking(ms3ggt_1),250).
orientation(ms3ggt_0,154,251). coords(ms3ggt_0,58,154,251).  visible(ms3ggt_0,251). happensAt(inactive(ms3ggt_0),251). orientation(ms3ggt_1,167,251). coords(ms3ggt_1,53,165,251).  visible(ms3ggt_1,251). happensAt(walking(ms3ggt_1),251).
orientation(ms3ggt_0,154,252). coords(ms3ggt_0,58,154,252).  visible(ms3ggt_0,252). happensAt(inactive(ms3ggt_0),252). orientation(ms3ggt_1,167,252). coords(ms3ggt_1,53,165,252).  visible(ms3ggt_1,252). happensAt(walking(ms3ggt_1),252).
orientation(ms3ggt_0,154,253). coords(ms3ggt_0,58,153,253).  visible(ms3ggt_0,253). happensAt(walking(ms3ggt_0),253). orientation(ms3ggt_1,167,253). coords(ms3ggt_1,54,165,253).  visible(ms3ggt_1,253). happensAt(walking(ms3ggt_1),253).
orientation(ms3ggt_0,154,254). coords(ms3ggt_0,58,153,254).  visible(ms3ggt_0,254). happensAt(walking(ms3ggt_0),254). orientation(ms3ggt_1,167,254). coords(ms3ggt_1,55,165,254).  visible(ms3ggt_1,254). happensAt(walking(ms3ggt_1),254).
#show holdsAt/2.