
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

	orientation(fra1gt_6,109,234). coords(fra1gt_6,251,22,234).  visible(fra1gt_6,234). happensAt(walking(fra1gt_6),234). orientation(fra1gt_7,127,234). coords(fra1gt_7,156,36,234).  visible(fra1gt_7,234). happensAt(walking(fra1gt_7),234).
orientation(fra1gt_6,109,235). coords(fra1gt_6,250,23,235).  visible(fra1gt_6,235). happensAt(walking(fra1gt_6),235). orientation(fra1gt_7,127,235). coords(fra1gt_7,156,36,235).  visible(fra1gt_7,235). happensAt(walking(fra1gt_7),235).
orientation(fra1gt_6,109,236). coords(fra1gt_6,250,26,236).  visible(fra1gt_6,236). happensAt(walking(fra1gt_6),236). orientation(fra1gt_7,132,236). coords(fra1gt_7,156,36,236).  visible(fra1gt_7,236). happensAt(walking(fra1gt_7),236).
orientation(fra1gt_6,109,237). coords(fra1gt_6,249,28,237).  visible(fra1gt_6,237). happensAt(walking(fra1gt_6),237). orientation(fra1gt_7,132,237). coords(fra1gt_7,156,36,237).  visible(fra1gt_7,237). happensAt(walking(fra1gt_7),237).
orientation(fra1gt_6,114,238). coords(fra1gt_6,248,30,238).  visible(fra1gt_6,238). happensAt(walking(fra1gt_6),238). orientation(fra1gt_7,137,238). coords(fra1gt_7,155,36,238).  visible(fra1gt_7,238). happensAt(walking(fra1gt_7),238).
orientation(fra1gt_6,114,239). coords(fra1gt_6,248,31,239).  visible(fra1gt_6,239). happensAt(walking(fra1gt_6),239). orientation(fra1gt_7,137,239). coords(fra1gt_7,155,36,239).  visible(fra1gt_7,239). happensAt(walking(fra1gt_7),239).
orientation(fra1gt_6,114,240). coords(fra1gt_6,247,33,240).  visible(fra1gt_6,240). happensAt(walking(fra1gt_6),240). orientation(fra1gt_7,137,240). coords(fra1gt_7,154,36,240).  visible(fra1gt_7,240). happensAt(walking(fra1gt_7),240).
orientation(fra1gt_6,114,241). coords(fra1gt_6,246,33,241).  visible(fra1gt_6,241). happensAt(walking(fra1gt_6),241). orientation(fra1gt_7,137,241). coords(fra1gt_7,154,36,241).  visible(fra1gt_7,241). happensAt(walking(fra1gt_7),241).
orientation(fra1gt_6,114,242). coords(fra1gt_6,245,34,242).  visible(fra1gt_6,242). happensAt(walking(fra1gt_6),242). orientation(fra1gt_7,137,242). coords(fra1gt_7,154,36,242).  visible(fra1gt_7,242). happensAt(walking(fra1gt_7),242).
orientation(fra1gt_6,119,243). coords(fra1gt_6,244,34,243).  visible(fra1gt_6,243). happensAt(walking(fra1gt_6),243). orientation(fra1gt_7,137,243). coords(fra1gt_7,153,37,243).  visible(fra1gt_7,243). happensAt(walking(fra1gt_7),243).
orientation(fra1gt_6,119,244). coords(fra1gt_6,243,35,244).  visible(fra1gt_6,244). happensAt(walking(fra1gt_6),244). orientation(fra1gt_7,137,244). coords(fra1gt_7,153,37,244).  visible(fra1gt_7,244). happensAt(walking(fra1gt_7),244).
orientation(fra1gt_6,123,245). coords(fra1gt_6,241,35,245).  visible(fra1gt_6,245). happensAt(walking(fra1gt_6),245). orientation(fra1gt_7,137,245). coords(fra1gt_7,151,37,245).  visible(fra1gt_7,245). happensAt(walking(fra1gt_7),245).
orientation(fra1gt_6,123,246). coords(fra1gt_6,240,35,246).  visible(fra1gt_6,246). happensAt(walking(fra1gt_6),246). orientation(fra1gt_7,137,246). coords(fra1gt_7,151,37,246).  visible(fra1gt_7,246). happensAt(walking(fra1gt_7),246).
orientation(fra1gt_6,123,247). coords(fra1gt_6,238,36,247).  visible(fra1gt_6,247). happensAt(walking(fra1gt_6),247). orientation(fra1gt_7,137,247). coords(fra1gt_7,151,37,247).  visible(fra1gt_7,247). happensAt(walking(fra1gt_7),247).
orientation(fra1gt_6,123,248). coords(fra1gt_6,236,37,248).  visible(fra1gt_6,248). happensAt(walking(fra1gt_6),248). orientation(fra1gt_7,137,248). coords(fra1gt_7,151,37,248).  visible(fra1gt_7,248). happensAt(walking(fra1gt_7),248).
orientation(fra1gt_6,123,249). coords(fra1gt_6,235,40,249).  visible(fra1gt_6,249). happensAt(walking(fra1gt_6),249). orientation(fra1gt_7,137,249). coords(fra1gt_7,151,38,249).  visible(fra1gt_7,249). happensAt(walking(fra1gt_7),249).
orientation(fra1gt_6,123,250). coords(fra1gt_6,234,43,250).  visible(fra1gt_6,250). happensAt(walking(fra1gt_6),250). orientation(fra1gt_7,137,250). coords(fra1gt_7,151,38,250).  visible(fra1gt_7,250). happensAt(walking(fra1gt_7),250).
orientation(fra1gt_6,123,251). coords(fra1gt_6,233,45,251).  visible(fra1gt_6,251). happensAt(walking(fra1gt_6),251). orientation(fra1gt_7,137,251). coords(fra1gt_7,150,39,251).  visible(fra1gt_7,251). happensAt(walking(fra1gt_7),251).
orientation(fra1gt_6,123,252). coords(fra1gt_6,233,46,252).  visible(fra1gt_6,252). happensAt(walking(fra1gt_6),252). orientation(fra1gt_7,137,252). coords(fra1gt_7,150,39,252).  visible(fra1gt_7,252). happensAt(walking(fra1gt_7),252).
orientation(fra1gt_6,123,253). coords(fra1gt_6,232,48,253).  visible(fra1gt_6,253). happensAt(walking(fra1gt_6),253). orientation(fra1gt_7,137,253). coords(fra1gt_7,150,39,253).  visible(fra1gt_7,253). happensAt(walking(fra1gt_7),253).
orientation(fra1gt_6,123,254). coords(fra1gt_6,232,49,254).  visible(fra1gt_6,254). happensAt(walking(fra1gt_6),254). orientation(fra1gt_7,137,254). coords(fra1gt_7,149,40,254).  visible(fra1gt_7,254). happensAt(walking(fra1gt_7),254).
orientation(fra1gt_6,123,255). coords(fra1gt_6,232,49,255).  visible(fra1gt_6,255). happensAt(walking(fra1gt_6),255). orientation(fra1gt_7,137,255). coords(fra1gt_7,149,40,255).  visible(fra1gt_7,255). happensAt(walking(fra1gt_7),255).
orientation(fra1gt_6,123,256). coords(fra1gt_6,231,50,256).  visible(fra1gt_6,256). happensAt(walking(fra1gt_6),256). orientation(fra1gt_7,137,256). coords(fra1gt_7,149,40,256).  visible(fra1gt_7,256). happensAt(walking(fra1gt_7),256).
orientation(fra1gt_6,123,257). coords(fra1gt_6,230,51,257).  visible(fra1gt_6,257). happensAt(walking(fra1gt_6),257). orientation(fra1gt_7,137,257). coords(fra1gt_7,148,41,257).  visible(fra1gt_7,257). happensAt(walking(fra1gt_7),257).
orientation(fra1gt_6,123,258). coords(fra1gt_6,228,51,258).  visible(fra1gt_6,258). happensAt(walking(fra1gt_6),258). orientation(fra1gt_7,137,258). coords(fra1gt_7,148,41,258).  visible(fra1gt_7,258). happensAt(walking(fra1gt_7),258).
#show holdsAt/2.