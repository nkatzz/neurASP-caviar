
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

	orientation(mc1gt_2,147,228). coords(mc1gt_2,123,129,228).  visible(mc1gt_2,228). happensAt(walking(mc1gt_2),228). orientation(mc1gt_3,146,228). coords(mc1gt_3,165,88,228).  visible(mc1gt_3,228). happensAt(walking(mc1gt_3),228).
orientation(mc1gt_2,147,229). coords(mc1gt_2,124,130,229).  visible(mc1gt_2,229). happensAt(walking(mc1gt_2),229). orientation(mc1gt_3,146,229). coords(mc1gt_3,167,88,229).  visible(mc1gt_3,229). happensAt(walking(mc1gt_3),229).
orientation(mc1gt_2,147,230). coords(mc1gt_2,125,131,230).  visible(mc1gt_2,230). happensAt(walking(mc1gt_2),230). orientation(mc1gt_3,143,230). coords(mc1gt_3,169,88,230).  visible(mc1gt_3,230). happensAt(walking(mc1gt_3),230).
orientation(mc1gt_2,146,231). coords(mc1gt_2,127,132,231).  visible(mc1gt_2,231). happensAt(walking(mc1gt_2),231). orientation(mc1gt_3,141,231). coords(mc1gt_3,171,88,231).  visible(mc1gt_3,231). happensAt(walking(mc1gt_3),231).
orientation(mc1gt_2,145,232). coords(mc1gt_2,129,133,232).  visible(mc1gt_2,232). happensAt(walking(mc1gt_2),232). orientation(mc1gt_3,139,232). coords(mc1gt_3,172,88,232).  visible(mc1gt_3,232). happensAt(walking(mc1gt_3),232).
orientation(mc1gt_2,147,233). coords(mc1gt_2,130,134,233).  visible(mc1gt_2,233). happensAt(walking(mc1gt_2),233). orientation(mc1gt_3,141,233). coords(mc1gt_3,173,88,233).  visible(mc1gt_3,233). happensAt(walking(mc1gt_3),233).
orientation(mc1gt_2,147,234). coords(mc1gt_2,131,134,234).  visible(mc1gt_2,234). happensAt(walking(mc1gt_2),234). orientation(mc1gt_3,140,234). coords(mc1gt_3,173,88,234).  visible(mc1gt_3,234). happensAt(walking(mc1gt_3),234).
orientation(mc1gt_2,147,235). coords(mc1gt_2,131,135,235).  visible(mc1gt_2,235). happensAt(walking(mc1gt_2),235). orientation(mc1gt_3,141,235). coords(mc1gt_3,174,89,235).  visible(mc1gt_3,235). happensAt(walking(mc1gt_3),235).
orientation(mc1gt_2,146,236). coords(mc1gt_2,131,135,236).  visible(mc1gt_2,236). happensAt(walking(mc1gt_2),236). orientation(mc1gt_3,138,236). coords(mc1gt_3,174,89,236).  visible(mc1gt_3,236). happensAt(walking(mc1gt_3),236).
orientation(mc1gt_2,146,237). coords(mc1gt_2,132,135,237).  visible(mc1gt_2,237). happensAt(walking(mc1gt_2),237). orientation(mc1gt_3,137,237). coords(mc1gt_3,175,90,237).  visible(mc1gt_3,237). happensAt(walking(mc1gt_3),237).
orientation(mc1gt_2,146,238). coords(mc1gt_2,132,136,238).  visible(mc1gt_2,238). happensAt(walking(mc1gt_2),238). orientation(mc1gt_3,137,238). coords(mc1gt_3,176,92,238).  visible(mc1gt_3,238). happensAt(walking(mc1gt_3),238).
orientation(mc1gt_2,146,239). coords(mc1gt_2,132,136,239).  visible(mc1gt_2,239). happensAt(walking(mc1gt_2),239). orientation(mc1gt_3,139,239). coords(mc1gt_3,178,93,239).  visible(mc1gt_3,239). happensAt(walking(mc1gt_3),239).
orientation(mc1gt_2,146,240). coords(mc1gt_2,133,137,240).  visible(mc1gt_2,240). happensAt(walking(mc1gt_2),240). orientation(mc1gt_3,139,240). coords(mc1gt_3,181,94,240).  visible(mc1gt_3,240). happensAt(walking(mc1gt_3),240).
orientation(mc1gt_2,146,241). coords(mc1gt_2,133,137,241).  visible(mc1gt_2,241). happensAt(walking(mc1gt_2),241). orientation(mc1gt_3,139,241). coords(mc1gt_3,183,95,241).  visible(mc1gt_3,241). happensAt(walking(mc1gt_3),241).
orientation(mc1gt_2,146,242). coords(mc1gt_2,133,137,242).  visible(mc1gt_2,242). happensAt(walking(mc1gt_2),242). orientation(mc1gt_3,139,242). coords(mc1gt_3,185,95,242).  visible(mc1gt_3,242). happensAt(walking(mc1gt_3),242).
orientation(mc1gt_2,146,243). coords(mc1gt_2,134,137,243).  visible(mc1gt_2,243). happensAt(walking(mc1gt_2),243). orientation(mc1gt_3,139,243). coords(mc1gt_3,187,96,243).  visible(mc1gt_3,243). happensAt(walking(mc1gt_3),243).
orientation(mc1gt_2,146,244). coords(mc1gt_2,134,138,244).  visible(mc1gt_2,244). happensAt(walking(mc1gt_2),244). orientation(mc1gt_3,139,244). coords(mc1gt_3,188,97,244).  visible(mc1gt_3,244). happensAt(walking(mc1gt_3),244).
orientation(mc1gt_2,146,245). coords(mc1gt_2,135,138,245).  visible(mc1gt_2,245). happensAt(walking(mc1gt_2),245). orientation(mc1gt_3,139,245). coords(mc1gt_3,189,97,245).  visible(mc1gt_3,245). happensAt(walking(mc1gt_3),245).
orientation(mc1gt_2,146,246). coords(mc1gt_2,135,139,246).  visible(mc1gt_2,246). happensAt(walking(mc1gt_2),246). orientation(mc1gt_3,139,246). coords(mc1gt_3,190,97,246).  visible(mc1gt_3,246). happensAt(walking(mc1gt_3),246).
orientation(mc1gt_2,146,247). coords(mc1gt_2,136,139,247).  visible(mc1gt_2,247). happensAt(walking(mc1gt_2),247). orientation(mc1gt_3,139,247). coords(mc1gt_3,190,98,247).  visible(mc1gt_3,247). happensAt(walking(mc1gt_3),247).
orientation(mc1gt_2,146,248). coords(mc1gt_2,137,140,248).  visible(mc1gt_2,248). happensAt(walking(mc1gt_2),248). orientation(mc1gt_3,139,248). coords(mc1gt_3,191,98,248).  visible(mc1gt_3,248). happensAt(walking(mc1gt_3),248).
orientation(mc1gt_2,146,249). coords(mc1gt_2,137,140,249).  visible(mc1gt_2,249). happensAt(walking(mc1gt_2),249). orientation(mc1gt_3,139,249). coords(mc1gt_3,191,98,249).  visible(mc1gt_3,249). happensAt(walking(mc1gt_3),249).
orientation(mc1gt_2,146,250). coords(mc1gt_2,138,142,250).  visible(mc1gt_2,250). happensAt(walking(mc1gt_2),250). orientation(mc1gt_3,139,250). coords(mc1gt_3,192,98,250).  visible(mc1gt_3,250). happensAt(walking(mc1gt_3),250).
orientation(mc1gt_2,146,251). coords(mc1gt_2,138,142,251).  visible(mc1gt_2,251). happensAt(walking(mc1gt_2),251). orientation(mc1gt_3,139,251). coords(mc1gt_3,195,98,251).  visible(mc1gt_3,251). happensAt(walking(mc1gt_3),251).
orientation(mc1gt_2,146,252). coords(mc1gt_2,139,142,252).  visible(mc1gt_2,252). happensAt(walking(mc1gt_2),252). orientation(mc1gt_3,139,252). coords(mc1gt_3,199,99,252).  visible(mc1gt_3,252). happensAt(walking(mc1gt_3),252).
#show holdsAt/2.