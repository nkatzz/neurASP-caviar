
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

	orientation(mc1gt_1,142,228). coords(mc1gt_1,179,126,228).  visible(mc1gt_1,228). happensAt(walking(mc1gt_1),228). orientation(mc1gt_3,146,228). coords(mc1gt_3,165,88,228).  visible(mc1gt_3,228). happensAt(walking(mc1gt_3),228).
orientation(mc1gt_1,142,229). coords(mc1gt_1,181,129,229).  visible(mc1gt_1,229). happensAt(walking(mc1gt_1),229). orientation(mc1gt_3,146,229). coords(mc1gt_3,167,88,229).  visible(mc1gt_3,229). happensAt(walking(mc1gt_3),229).
orientation(mc1gt_1,142,230). coords(mc1gt_1,183,131,230).  visible(mc1gt_1,230). happensAt(walking(mc1gt_1),230). orientation(mc1gt_3,143,230). coords(mc1gt_3,169,88,230).  visible(mc1gt_3,230). happensAt(walking(mc1gt_3),230).
orientation(mc1gt_1,142,231). coords(mc1gt_1,184,133,231).  visible(mc1gt_1,231). happensAt(walking(mc1gt_1),231). orientation(mc1gt_3,141,231). coords(mc1gt_3,171,88,231).  visible(mc1gt_3,231). happensAt(walking(mc1gt_3),231).
orientation(mc1gt_1,142,232). coords(mc1gt_1,185,135,232).  visible(mc1gt_1,232). happensAt(walking(mc1gt_1),232). orientation(mc1gt_3,139,232). coords(mc1gt_3,172,88,232).  visible(mc1gt_3,232). happensAt(walking(mc1gt_3),232).
orientation(mc1gt_1,142,233). coords(mc1gt_1,186,137,233).  visible(mc1gt_1,233). happensAt(walking(mc1gt_1),233). orientation(mc1gt_3,141,233). coords(mc1gt_3,173,88,233).  visible(mc1gt_3,233). happensAt(walking(mc1gt_3),233).
orientation(mc1gt_1,142,234). coords(mc1gt_1,187,137,234).  visible(mc1gt_1,234). happensAt(walking(mc1gt_1),234). orientation(mc1gt_3,140,234). coords(mc1gt_3,173,88,234).  visible(mc1gt_3,234). happensAt(walking(mc1gt_3),234).
orientation(mc1gt_1,142,235). coords(mc1gt_1,188,138,235).  visible(mc1gt_1,235). happensAt(walking(mc1gt_1),235). orientation(mc1gt_3,141,235). coords(mc1gt_3,174,89,235).  visible(mc1gt_3,235). happensAt(walking(mc1gt_3),235).
orientation(mc1gt_1,142,236). coords(mc1gt_1,189,139,236).  visible(mc1gt_1,236). happensAt(walking(mc1gt_1),236). orientation(mc1gt_3,138,236). coords(mc1gt_3,174,89,236).  visible(mc1gt_3,236). happensAt(walking(mc1gt_3),236).
orientation(mc1gt_1,142,237). coords(mc1gt_1,189,140,237).  visible(mc1gt_1,237). happensAt(walking(mc1gt_1),237). orientation(mc1gt_3,137,237). coords(mc1gt_3,175,90,237).  visible(mc1gt_3,237). happensAt(walking(mc1gt_3),237).
orientation(mc1gt_1,142,238). coords(mc1gt_1,190,141,238).  visible(mc1gt_1,238). happensAt(walking(mc1gt_1),238). orientation(mc1gt_3,137,238). coords(mc1gt_3,176,92,238).  visible(mc1gt_3,238). happensAt(walking(mc1gt_3),238).
orientation(mc1gt_1,142,239). coords(mc1gt_1,190,141,239).  visible(mc1gt_1,239). happensAt(walking(mc1gt_1),239). orientation(mc1gt_3,139,239). coords(mc1gt_3,178,93,239).  visible(mc1gt_3,239). happensAt(walking(mc1gt_3),239).
orientation(mc1gt_1,145,240). coords(mc1gt_1,190,142,240).  visible(mc1gt_1,240). happensAt(walking(mc1gt_1),240). orientation(mc1gt_3,139,240). coords(mc1gt_3,181,94,240).  visible(mc1gt_3,240). happensAt(walking(mc1gt_3),240).
orientation(mc1gt_1,147,241). coords(mc1gt_1,191,142,241).  visible(mc1gt_1,241). happensAt(walking(mc1gt_1),241). orientation(mc1gt_3,139,241). coords(mc1gt_3,183,95,241).  visible(mc1gt_3,241). happensAt(walking(mc1gt_3),241).
orientation(mc1gt_1,146,242). coords(mc1gt_1,192,143,242).  visible(mc1gt_1,242). happensAt(walking(mc1gt_1),242). orientation(mc1gt_3,139,242). coords(mc1gt_3,185,95,242).  visible(mc1gt_3,242). happensAt(walking(mc1gt_3),242).
orientation(mc1gt_1,148,243). coords(mc1gt_1,194,144,243).  visible(mc1gt_1,243). happensAt(walking(mc1gt_1),243). orientation(mc1gt_3,139,243). coords(mc1gt_3,187,96,243).  visible(mc1gt_3,243). happensAt(walking(mc1gt_3),243).
orientation(mc1gt_1,145,244). coords(mc1gt_1,197,145,244).  visible(mc1gt_1,244). happensAt(walking(mc1gt_1),244). orientation(mc1gt_3,139,244). coords(mc1gt_3,188,97,244).  visible(mc1gt_3,244). happensAt(walking(mc1gt_3),244).
orientation(mc1gt_1,142,245). coords(mc1gt_1,198,149,245).  visible(mc1gt_1,245). happensAt(walking(mc1gt_1),245). orientation(mc1gt_3,139,245). coords(mc1gt_3,189,97,245).  visible(mc1gt_3,245). happensAt(walking(mc1gt_3),245).
orientation(mc1gt_1,142,246). coords(mc1gt_1,200,151,246).  visible(mc1gt_1,246). happensAt(walking(mc1gt_1),246). orientation(mc1gt_3,139,246). coords(mc1gt_3,190,97,246).  visible(mc1gt_3,246). happensAt(walking(mc1gt_3),246).
orientation(mc1gt_1,142,247). coords(mc1gt_1,201,154,247).  visible(mc1gt_1,247). happensAt(walking(mc1gt_1),247). orientation(mc1gt_3,139,247). coords(mc1gt_3,190,98,247).  visible(mc1gt_3,247). happensAt(walking(mc1gt_3),247).
orientation(mc1gt_1,142,248). coords(mc1gt_1,202,155,248).  visible(mc1gt_1,248). happensAt(walking(mc1gt_1),248). orientation(mc1gt_3,139,248). coords(mc1gt_3,191,98,248).  visible(mc1gt_3,248). happensAt(walking(mc1gt_3),248).
orientation(mc1gt_1,142,249). coords(mc1gt_1,203,157,249).  visible(mc1gt_1,249). happensAt(walking(mc1gt_1),249). orientation(mc1gt_3,139,249). coords(mc1gt_3,191,98,249).  visible(mc1gt_3,249). happensAt(walking(mc1gt_3),249).
orientation(mc1gt_1,142,250). coords(mc1gt_1,204,158,250).  visible(mc1gt_1,250). happensAt(walking(mc1gt_1),250). orientation(mc1gt_3,139,250). coords(mc1gt_3,192,98,250).  visible(mc1gt_3,250). happensAt(walking(mc1gt_3),250).
orientation(mc1gt_1,142,251). coords(mc1gt_1,205,159,251).  visible(mc1gt_1,251). happensAt(walking(mc1gt_1),251). orientation(mc1gt_3,139,251). coords(mc1gt_3,195,98,251).  visible(mc1gt_3,251). happensAt(walking(mc1gt_3),251).
orientation(mc1gt_1,144,252). coords(mc1gt_1,205,159,252).  visible(mc1gt_1,252). happensAt(walking(mc1gt_1),252). orientation(mc1gt_3,139,252). coords(mc1gt_3,199,99,252).  visible(mc1gt_3,252). happensAt(walking(mc1gt_3),252).
#show holdsAt/2.