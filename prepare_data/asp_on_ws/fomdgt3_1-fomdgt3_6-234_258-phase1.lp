
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

	orientation(fomdgt3_1,150,234). coords(fomdgt3_1,68,93,234).  visible(fomdgt3_1,234). happensAt(walking(fomdgt3_1),234). orientation(fomdgt3_6,130,234). coords(fomdgt3_6,216,104,234).  visible(fomdgt3_6,234). happensAt(walking(fomdgt3_6),234).
orientation(fomdgt3_1,150,235). coords(fomdgt3_1,66,93,235).  visible(fomdgt3_1,235). happensAt(walking(fomdgt3_1),235). orientation(fomdgt3_6,130,235). coords(fomdgt3_6,216,104,235).  visible(fomdgt3_6,235). happensAt(walking(fomdgt3_6),235).
orientation(fomdgt3_1,150,236). coords(fomdgt3_1,66,93,236).  visible(fomdgt3_1,236). happensAt(walking(fomdgt3_1),236). orientation(fomdgt3_6,130,236). coords(fomdgt3_6,212,104,236).  visible(fomdgt3_6,236). happensAt(walking(fomdgt3_6),236).
orientation(fomdgt3_1,150,237). coords(fomdgt3_1,66,93,237).  visible(fomdgt3_1,237). happensAt(walking(fomdgt3_1),237). orientation(fomdgt3_6,130,237). coords(fomdgt3_6,209,106,237).  visible(fomdgt3_6,237). happensAt(walking(fomdgt3_6),237).
orientation(fomdgt3_1,150,238). coords(fomdgt3_1,66,93,238).  visible(fomdgt3_1,238). happensAt(walking(fomdgt3_1),238). orientation(fomdgt3_6,130,238). coords(fomdgt3_6,209,104,238).  visible(fomdgt3_6,238). happensAt(walking(fomdgt3_6),238).
orientation(fomdgt3_1,150,239). coords(fomdgt3_1,66,93,239).  visible(fomdgt3_1,239). happensAt(walking(fomdgt3_1),239). orientation(fomdgt3_6,130,239). coords(fomdgt3_6,209,104,239).  visible(fomdgt3_6,239). happensAt(walking(fomdgt3_6),239).
orientation(fomdgt3_1,150,240). coords(fomdgt3_1,66,93,240).  visible(fomdgt3_1,240). happensAt(walking(fomdgt3_1),240). orientation(fomdgt3_6,130,240). coords(fomdgt3_6,209,111,240).  visible(fomdgt3_6,240). happensAt(walking(fomdgt3_6),240).
orientation(fomdgt3_1,150,241). coords(fomdgt3_1,64,93,241).  visible(fomdgt3_1,241). happensAt(walking(fomdgt3_1),241). orientation(fomdgt3_6,130,241). coords(fomdgt3_6,213,114,241).  visible(fomdgt3_6,241). happensAt(walking(fomdgt3_6),241).
orientation(fomdgt3_1,150,242). coords(fomdgt3_1,64,93,242).  visible(fomdgt3_1,242). happensAt(walking(fomdgt3_1),242). orientation(fomdgt3_6,130,242). coords(fomdgt3_6,215,120,242).  visible(fomdgt3_6,242). happensAt(walking(fomdgt3_6),242).
orientation(fomdgt3_1,150,243). coords(fomdgt3_1,64,93,243).  visible(fomdgt3_1,243). happensAt(walking(fomdgt3_1),243). orientation(fomdgt3_6,130,243). coords(fomdgt3_6,215,122,243).  visible(fomdgt3_6,243). happensAt(walking(fomdgt3_6),243).
orientation(fomdgt3_1,150,244). coords(fomdgt3_1,64,93,244).  visible(fomdgt3_1,244). happensAt(walking(fomdgt3_1),244). orientation(fomdgt3_6,130,244). coords(fomdgt3_6,210,129,244).  visible(fomdgt3_6,244). happensAt(walking(fomdgt3_6),244).
orientation(fomdgt3_1,150,245). coords(fomdgt3_1,64,93,245).  visible(fomdgt3_1,245). happensAt(walking(fomdgt3_1),245). orientation(fomdgt3_6,130,245). coords(fomdgt3_6,210,129,245).  visible(fomdgt3_6,245). happensAt(walking(fomdgt3_6),245).
orientation(fomdgt3_1,150,246). coords(fomdgt3_1,64,93,246).  visible(fomdgt3_1,246). happensAt(walking(fomdgt3_1),246). orientation(fomdgt3_6,130,246). coords(fomdgt3_6,209,133,246).  visible(fomdgt3_6,246). happensAt(walking(fomdgt3_6),246).
orientation(fomdgt3_1,150,247). coords(fomdgt3_1,64,93,247).  visible(fomdgt3_1,247). happensAt(walking(fomdgt3_1),247). orientation(fomdgt3_6,130,247). coords(fomdgt3_6,209,135,247).  visible(fomdgt3_6,247). happensAt(walking(fomdgt3_6),247).
orientation(fomdgt3_1,150,248). coords(fomdgt3_1,64,93,248).  visible(fomdgt3_1,248). happensAt(walking(fomdgt3_1),248). orientation(fomdgt3_6,130,248). coords(fomdgt3_6,209,135,248).  visible(fomdgt3_6,248). happensAt(walking(fomdgt3_6),248).
orientation(fomdgt3_1,150,249). coords(fomdgt3_1,61,94,249).  visible(fomdgt3_1,249). happensAt(walking(fomdgt3_1),249). orientation(fomdgt3_6,130,249). coords(fomdgt3_6,203,137,249).  visible(fomdgt3_6,249). happensAt(walking(fomdgt3_6),249).
orientation(fomdgt3_1,150,250). coords(fomdgt3_1,61,94,250).  visible(fomdgt3_1,250). happensAt(walking(fomdgt3_1),250). orientation(fomdgt3_6,130,250). coords(fomdgt3_6,203,137,250).  visible(fomdgt3_6,250). happensAt(walking(fomdgt3_6),250).
orientation(fomdgt3_1,150,251). coords(fomdgt3_1,61,94,251).  visible(fomdgt3_1,251). happensAt(walking(fomdgt3_1),251). orientation(fomdgt3_6,130,251). coords(fomdgt3_6,203,140,251).  visible(fomdgt3_6,251). happensAt(walking(fomdgt3_6),251).
orientation(fomdgt3_1,150,252). coords(fomdgt3_1,61,94,252).  visible(fomdgt3_1,252). happensAt(walking(fomdgt3_1),252). orientation(fomdgt3_6,130,252). coords(fomdgt3_6,203,142,252).  visible(fomdgt3_6,252). happensAt(walking(fomdgt3_6),252).
orientation(fomdgt3_1,150,253). coords(fomdgt3_1,61,94,253).  visible(fomdgt3_1,253). happensAt(walking(fomdgt3_1),253). orientation(fomdgt3_6,130,253). coords(fomdgt3_6,203,144,253).  visible(fomdgt3_6,253). happensAt(walking(fomdgt3_6),253).
orientation(fomdgt3_1,150,254). coords(fomdgt3_1,61,94,254).  visible(fomdgt3_1,254). happensAt(walking(fomdgt3_1),254). orientation(fomdgt3_6,130,254). coords(fomdgt3_6,203,140,254).  visible(fomdgt3_6,254). happensAt(walking(fomdgt3_6),254).
orientation(fomdgt3_1,150,255). coords(fomdgt3_1,61,94,255).  visible(fomdgt3_1,255). happensAt(walking(fomdgt3_1),255). orientation(fomdgt3_6,130,255). coords(fomdgt3_6,205,146,255).  visible(fomdgt3_6,255). happensAt(walking(fomdgt3_6),255).
orientation(fomdgt3_1,150,256). coords(fomdgt3_1,61,94,256).  visible(fomdgt3_1,256). happensAt(walking(fomdgt3_1),256). orientation(fomdgt3_6,130,256). coords(fomdgt3_6,205,146,256).  visible(fomdgt3_6,256). happensAt(walking(fomdgt3_6),256).
orientation(fomdgt3_1,150,257). coords(fomdgt3_1,61,94,257).  visible(fomdgt3_1,257). happensAt(walking(fomdgt3_1),257). orientation(fomdgt3_6,130,257). coords(fomdgt3_6,206,150,257).  visible(fomdgt3_6,257). happensAt(walking(fomdgt3_6),257).
orientation(fomdgt3_1,150,258). coords(fomdgt3_1,61,94,258).  visible(fomdgt3_1,258). happensAt(walking(fomdgt3_1),258). orientation(fomdgt3_6,130,258). coords(fomdgt3_6,206,150,258).  visible(fomdgt3_6,258). happensAt(walking(fomdgt3_6),258).
#show holdsAt/2.