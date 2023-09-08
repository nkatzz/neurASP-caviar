
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

	orientation(lbgt_0,158,228). coords(lbgt_0,167,139,228).  visible(lbgt_0,228). happensAt(walking(lbgt_0),228). orientation(lbgt_1,147,228). coords(lbgt_1,81,72,228).  visible(lbgt_1,228). happensAt(walking(lbgt_1),228).
orientation(lbgt_0,158,229). coords(lbgt_0,166,137,229).  visible(lbgt_0,229). happensAt(walking(lbgt_0),229). orientation(lbgt_1,147,229). coords(lbgt_1,81,72,229).  visible(lbgt_1,229). happensAt(walking(lbgt_1),229).
orientation(lbgt_0,158,230). coords(lbgt_0,163,137,230).  visible(lbgt_0,230). happensAt(walking(lbgt_0),230). orientation(lbgt_1,147,230). coords(lbgt_1,81,71,230).  visible(lbgt_1,230). happensAt(walking(lbgt_1),230).
orientation(lbgt_0,158,231). coords(lbgt_0,162,136,231).  visible(lbgt_0,231). happensAt(walking(lbgt_0),231). orientation(lbgt_1,147,231). coords(lbgt_1,81,70,231).  visible(lbgt_1,231). happensAt(walking(lbgt_1),231).
orientation(lbgt_0,158,232). coords(lbgt_0,160,136,232).  visible(lbgt_0,232). happensAt(walking(lbgt_0),232). orientation(lbgt_1,147,232). coords(lbgt_1,81,69,232).  visible(lbgt_1,232). happensAt(walking(lbgt_1),232).
orientation(lbgt_0,158,233). coords(lbgt_0,159,135,233).  visible(lbgt_0,233). happensAt(walking(lbgt_0),233). orientation(lbgt_1,147,233). coords(lbgt_1,81,68,233).  visible(lbgt_1,233). happensAt(walking(lbgt_1),233).
orientation(lbgt_0,152,234). coords(lbgt_0,158,135,234).  visible(lbgt_0,234). happensAt(walking(lbgt_0),234). orientation(lbgt_1,147,234). coords(lbgt_1,81,68,234).  visible(lbgt_1,234). happensAt(walking(lbgt_1),234).
orientation(lbgt_0,152,235). coords(lbgt_0,158,135,235).  visible(lbgt_0,235). happensAt(walking(lbgt_0),235). orientation(lbgt_1,147,235). coords(lbgt_1,82,68,235).  visible(lbgt_1,235). happensAt(walking(lbgt_1),235).
orientation(lbgt_0,152,236). coords(lbgt_0,158,134,236).  visible(lbgt_0,236). happensAt(walking(lbgt_0),236). orientation(lbgt_1,147,236). coords(lbgt_1,82,68,236).  visible(lbgt_1,236). happensAt(walking(lbgt_1),236).
orientation(lbgt_0,152,237). coords(lbgt_0,158,134,237).  visible(lbgt_0,237). happensAt(walking(lbgt_0),237). orientation(lbgt_1,147,237). coords(lbgt_1,82,68,237).  visible(lbgt_1,237). happensAt(walking(lbgt_1),237).
orientation(lbgt_0,152,238). coords(lbgt_0,157,133,238).  visible(lbgt_0,238). happensAt(walking(lbgt_0),238). orientation(lbgt_1,147,238). coords(lbgt_1,82,68,238).  visible(lbgt_1,238). happensAt(walking(lbgt_1),238).
orientation(lbgt_0,152,239). coords(lbgt_0,157,133,239).  visible(lbgt_0,239). happensAt(walking(lbgt_0),239). orientation(lbgt_1,147,239). coords(lbgt_1,82,68,239).  visible(lbgt_1,239). happensAt(walking(lbgt_1),239).
orientation(lbgt_0,148,240). coords(lbgt_0,156,132,240).  visible(lbgt_0,240). happensAt(walking(lbgt_0),240). orientation(lbgt_1,147,240). coords(lbgt_1,82,68,240).  visible(lbgt_1,240). happensAt(walking(lbgt_1),240).
orientation(lbgt_0,148,241). coords(lbgt_0,155,131,241).  visible(lbgt_0,241). happensAt(walking(lbgt_0),241). orientation(lbgt_1,147,241). coords(lbgt_1,82,67,241).  visible(lbgt_1,241). happensAt(walking(lbgt_1),241).
orientation(lbgt_0,148,242). coords(lbgt_0,154,130,242).  visible(lbgt_0,242). happensAt(walking(lbgt_0),242). orientation(lbgt_1,147,242). coords(lbgt_1,82,67,242).  visible(lbgt_1,242). happensAt(walking(lbgt_1),242).
orientation(lbgt_0,148,243). coords(lbgt_0,153,128,243).  visible(lbgt_0,243). happensAt(walking(lbgt_0),243). orientation(lbgt_1,147,243). coords(lbgt_1,82,65,243).  visible(lbgt_1,243). happensAt(walking(lbgt_1),243).
orientation(lbgt_0,148,244). coords(lbgt_0,152,127,244).  visible(lbgt_0,244). happensAt(walking(lbgt_0),244). orientation(lbgt_1,147,244). coords(lbgt_1,82,64,244).  visible(lbgt_1,244). happensAt(walking(lbgt_1),244).
orientation(lbgt_0,148,245). coords(lbgt_0,151,127,245).  visible(lbgt_0,245). happensAt(walking(lbgt_0),245). orientation(lbgt_1,147,245). coords(lbgt_1,82,64,245).  visible(lbgt_1,245). happensAt(walking(lbgt_1),245).
orientation(lbgt_0,148,246). coords(lbgt_0,151,126,246).  visible(lbgt_0,246). happensAt(walking(lbgt_0),246). orientation(lbgt_1,147,246). coords(lbgt_1,82,64,246).  visible(lbgt_1,246). happensAt(walking(lbgt_1),246).
orientation(lbgt_0,148,247). coords(lbgt_0,150,126,247).  visible(lbgt_0,247). happensAt(walking(lbgt_0),247). orientation(lbgt_1,147,247). coords(lbgt_1,81,63,247).  visible(lbgt_1,247). happensAt(walking(lbgt_1),247).
orientation(lbgt_0,148,248). coords(lbgt_0,150,126,248).  visible(lbgt_0,248). happensAt(walking(lbgt_0),248). orientation(lbgt_1,147,248). coords(lbgt_1,79,63,248).  visible(lbgt_1,248). happensAt(walking(lbgt_1),248).
orientation(lbgt_0,148,249). coords(lbgt_0,150,126,249).  visible(lbgt_0,249). happensAt(walking(lbgt_0),249). orientation(lbgt_1,147,249). coords(lbgt_1,78,63,249).  visible(lbgt_1,249). happensAt(walking(lbgt_1),249).
orientation(lbgt_0,148,250). coords(lbgt_0,150,126,250).  visible(lbgt_0,250). happensAt(walking(lbgt_0),250). orientation(lbgt_1,147,250). coords(lbgt_1,77,63,250).  visible(lbgt_1,250). happensAt(walking(lbgt_1),250).
orientation(lbgt_0,148,251). coords(lbgt_0,149,126,251).  visible(lbgt_0,251). happensAt(walking(lbgt_0),251). orientation(lbgt_1,147,251). coords(lbgt_1,77,63,251).  visible(lbgt_1,251). happensAt(walking(lbgt_1),251).
orientation(lbgt_0,148,252). coords(lbgt_0,147,126,252).  visible(lbgt_0,252). happensAt(walking(lbgt_0),252). orientation(lbgt_1,147,252). coords(lbgt_1,77,62,252).  visible(lbgt_1,252). happensAt(walking(lbgt_1),252).
#show holdsAt/2.