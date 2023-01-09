
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

	orientation(lb1gt_1,151,228). coords(lb1gt_1,135,158,228).  visible(lb1gt_1,228). happensAt(walking(lb1gt_1),228). orientation(lb1gt_2,147,228). coords(lb1gt_2,142,123,228).  visible(lb1gt_2,228). happensAt(walking(lb1gt_2),228).
orientation(lb1gt_1,151,229). coords(lb1gt_1,136,158,229).  visible(lb1gt_1,229). happensAt(walking(lb1gt_1),229). orientation(lb1gt_2,147,229). coords(lb1gt_2,144,123,229).  visible(lb1gt_2,229). happensAt(walking(lb1gt_2),229).
orientation(lb1gt_1,151,230). coords(lb1gt_1,136,158,230).  visible(lb1gt_1,230). happensAt(walking(lb1gt_1),230). orientation(lb1gt_2,147,230). coords(lb1gt_2,146,123,230).  visible(lb1gt_2,230). happensAt(walking(lb1gt_2),230).
orientation(lb1gt_1,151,231). coords(lb1gt_1,136,159,231).  visible(lb1gt_1,231). happensAt(walking(lb1gt_1),231). orientation(lb1gt_2,147,231). coords(lb1gt_2,147,123,231).  visible(lb1gt_2,231). happensAt(walking(lb1gt_2),231).
orientation(lb1gt_1,151,232). coords(lb1gt_1,137,159,232).  visible(lb1gt_1,232). happensAt(walking(lb1gt_1),232). orientation(lb1gt_2,147,232). coords(lb1gt_2,147,123,232).  visible(lb1gt_2,232). happensAt(walking(lb1gt_2),232).
orientation(lb1gt_1,151,233). coords(lb1gt_1,139,159,233).  visible(lb1gt_1,233). happensAt(walking(lb1gt_1),233). orientation(lb1gt_2,147,233). coords(lb1gt_2,148,123,233).  visible(lb1gt_2,233). happensAt(walking(lb1gt_2),233).
orientation(lb1gt_1,151,234). coords(lb1gt_1,141,159,234).  visible(lb1gt_1,234). happensAt(walking(lb1gt_1),234). orientation(lb1gt_2,147,234). coords(lb1gt_2,150,123,234).  visible(lb1gt_2,234). happensAt(walking(lb1gt_2),234).
orientation(lb1gt_1,151,235). coords(lb1gt_1,142,159,235).  visible(lb1gt_1,235). happensAt(walking(lb1gt_1),235). orientation(lb1gt_2,149,235). coords(lb1gt_2,150,123,235).  visible(lb1gt_2,235). happensAt(walking(lb1gt_2),235).
orientation(lb1gt_1,151,236). coords(lb1gt_1,144,159,236).  visible(lb1gt_1,236). happensAt(walking(lb1gt_1),236). orientation(lb1gt_2,149,236). coords(lb1gt_2,151,123,236).  visible(lb1gt_2,236). happensAt(walking(lb1gt_2),236).
orientation(lb1gt_1,151,237). coords(lb1gt_1,145,159,237).  visible(lb1gt_1,237). happensAt(walking(lb1gt_1),237). orientation(lb1gt_2,149,237). coords(lb1gt_2,151,123,237).  visible(lb1gt_2,237). happensAt(walking(lb1gt_2),237).
orientation(lb1gt_1,151,238). coords(lb1gt_1,145,159,238).  visible(lb1gt_1,238). happensAt(walking(lb1gt_1),238). orientation(lb1gt_2,149,238). coords(lb1gt_2,152,123,238).  visible(lb1gt_2,238). happensAt(walking(lb1gt_2),238).
orientation(lb1gt_1,151,239). coords(lb1gt_1,146,160,239).  visible(lb1gt_1,239). happensAt(walking(lb1gt_1),239). orientation(lb1gt_2,149,239). coords(lb1gt_2,152,123,239).  visible(lb1gt_2,239). happensAt(walking(lb1gt_2),239).
orientation(lb1gt_1,151,240). coords(lb1gt_1,146,160,240).  visible(lb1gt_1,240). happensAt(walking(lb1gt_1),240). orientation(lb1gt_2,149,240). coords(lb1gt_2,152,124,240).  visible(lb1gt_2,240). happensAt(walking(lb1gt_2),240).
orientation(lb1gt_1,151,241). coords(lb1gt_1,147,160,241).  visible(lb1gt_1,241). happensAt(walking(lb1gt_1),241). orientation(lb1gt_2,149,241). coords(lb1gt_2,153,124,241).  visible(lb1gt_2,241). happensAt(walking(lb1gt_2),241).
orientation(lb1gt_1,151,242). coords(lb1gt_1,147,160,242).  visible(lb1gt_1,242). happensAt(walking(lb1gt_1),242). orientation(lb1gt_2,149,242). coords(lb1gt_2,153,124,242).  visible(lb1gt_2,242). happensAt(walking(lb1gt_2),242).
orientation(lb1gt_1,151,243). coords(lb1gt_1,147,160,243).  visible(lb1gt_1,243). happensAt(walking(lb1gt_1),243). orientation(lb1gt_2,149,243). coords(lb1gt_2,154,125,243).  visible(lb1gt_2,243). happensAt(walking(lb1gt_2),243).
orientation(lb1gt_1,151,244). coords(lb1gt_1,147,160,244).  visible(lb1gt_1,244). happensAt(walking(lb1gt_1),244). orientation(lb1gt_2,149,244). coords(lb1gt_2,154,125,244).  visible(lb1gt_2,244). happensAt(walking(lb1gt_2),244).
orientation(lb1gt_1,151,245). coords(lb1gt_1,147,160,245).  visible(lb1gt_1,245). happensAt(walking(lb1gt_1),245). orientation(lb1gt_2,149,245). coords(lb1gt_2,154,126,245).  visible(lb1gt_2,245). happensAt(walking(lb1gt_2),245).
orientation(lb1gt_1,151,246). coords(lb1gt_1,147,160,246).  visible(lb1gt_1,246). happensAt(walking(lb1gt_1),246). orientation(lb1gt_2,149,246). coords(lb1gt_2,156,127,246).  visible(lb1gt_2,246). happensAt(walking(lb1gt_2),246).
orientation(lb1gt_1,151,247). coords(lb1gt_1,148,160,247).  visible(lb1gt_1,247). happensAt(walking(lb1gt_1),247). orientation(lb1gt_2,149,247). coords(lb1gt_2,157,128,247).  visible(lb1gt_2,247). happensAt(walking(lb1gt_2),247).
orientation(lb1gt_1,151,248). coords(lb1gt_1,148,160,248).  visible(lb1gt_1,248). happensAt(walking(lb1gt_1),248). orientation(lb1gt_2,149,248). coords(lb1gt_2,158,129,248).  visible(lb1gt_2,248). happensAt(walking(lb1gt_2),248).
orientation(lb1gt_1,151,249). coords(lb1gt_1,148,160,249).  visible(lb1gt_1,249). happensAt(walking(lb1gt_1),249). orientation(lb1gt_2,149,249). coords(lb1gt_2,159,130,249).  visible(lb1gt_2,249). happensAt(walking(lb1gt_2),249).
orientation(lb1gt_1,151,250). coords(lb1gt_1,148,160,250).  visible(lb1gt_1,250). happensAt(walking(lb1gt_1),250). orientation(lb1gt_2,149,250). coords(lb1gt_2,159,130,250).  visible(lb1gt_2,250). happensAt(walking(lb1gt_2),250).
orientation(lb1gt_1,151,251). coords(lb1gt_1,149,160,251).  visible(lb1gt_1,251). happensAt(walking(lb1gt_1),251). orientation(lb1gt_2,149,251). coords(lb1gt_2,160,130,251).  visible(lb1gt_2,251). happensAt(walking(lb1gt_2),251).
orientation(lb1gt_1,151,252). coords(lb1gt_1,150,160,252).  visible(lb1gt_1,252). happensAt(walking(lb1gt_1),252). orientation(lb1gt_2,149,252). coords(lb1gt_2,161,130,252).  visible(lb1gt_2,252). happensAt(walking(lb1gt_2),252).
#show holdsAt/2.