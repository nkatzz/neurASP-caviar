
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

	orientation(mwt1gt_1,160,233). coords(mwt1gt_1,58,183,233).  visible(mwt1gt_1,233). happensAt(walking(mwt1gt_1),233). orientation(mwt1gt_2,110,233). coords(mwt1gt_2,272,14,233).  visible(mwt1gt_2,233). happensAt(walking(mwt1gt_2),233).
orientation(mwt1gt_1,160,234). coords(mwt1gt_1,60,184,234).  visible(mwt1gt_1,234). happensAt(walking(mwt1gt_1),234). orientation(mwt1gt_2,110,234). coords(mwt1gt_2,272,14,234).  visible(mwt1gt_2,234). happensAt(walking(mwt1gt_2),234).
orientation(mwt1gt_1,160,235). coords(mwt1gt_1,62,184,235).  visible(mwt1gt_1,235). happensAt(walking(mwt1gt_1),235). orientation(mwt1gt_2,110,235). coords(mwt1gt_2,272,16,235).  visible(mwt1gt_2,235). happensAt(walking(mwt1gt_2),235).
orientation(mwt1gt_1,160,236). coords(mwt1gt_1,62,184,236).  visible(mwt1gt_1,236). happensAt(walking(mwt1gt_1),236). orientation(mwt1gt_2,110,236). coords(mwt1gt_2,272,17,236).  visible(mwt1gt_2,236). happensAt(walking(mwt1gt_2),236).
orientation(mwt1gt_1,160,237). coords(mwt1gt_1,62,184,237).  visible(mwt1gt_1,237). happensAt(walking(mwt1gt_1),237). orientation(mwt1gt_2,110,237). coords(mwt1gt_2,270,19,237).  visible(mwt1gt_2,237). happensAt(walking(mwt1gt_2),237).
orientation(mwt1gt_1,160,238). coords(mwt1gt_1,62,184,238).  visible(mwt1gt_1,238). happensAt(walking(mwt1gt_1),238). orientation(mwt1gt_2,110,238). coords(mwt1gt_2,270,19,238).  visible(mwt1gt_2,238). happensAt(walking(mwt1gt_2),238).
orientation(mwt1gt_1,160,239). coords(mwt1gt_1,62,184,239).  visible(mwt1gt_1,239). happensAt(walking(mwt1gt_1),239). orientation(mwt1gt_2,110,239). coords(mwt1gt_2,266,20,239).  visible(mwt1gt_2,239). happensAt(walking(mwt1gt_2),239).
orientation(mwt1gt_1,160,240). coords(mwt1gt_1,62,184,240).  visible(mwt1gt_1,240). happensAt(walking(mwt1gt_1),240). orientation(mwt1gt_2,110,240). coords(mwt1gt_2,261,21,240).  visible(mwt1gt_2,240). happensAt(walking(mwt1gt_2),240).
orientation(mwt1gt_1,160,241). coords(mwt1gt_1,62,184,241).  visible(mwt1gt_1,241). happensAt(walking(mwt1gt_1),241). orientation(mwt1gt_2,110,241). coords(mwt1gt_2,261,21,241).  visible(mwt1gt_2,241). happensAt(walking(mwt1gt_2),241).
orientation(mwt1gt_1,160,242). coords(mwt1gt_1,62,184,242).  visible(mwt1gt_1,242). happensAt(walking(mwt1gt_1),242). orientation(mwt1gt_2,110,242). coords(mwt1gt_2,258,22,242).  visible(mwt1gt_2,242). happensAt(walking(mwt1gt_2),242).
orientation(mwt1gt_1,160,243). coords(mwt1gt_1,64,184,243).  visible(mwt1gt_1,243). happensAt(walking(mwt1gt_1),243). orientation(mwt1gt_2,110,243). coords(mwt1gt_2,256,22,243).  visible(mwt1gt_2,243). happensAt(walking(mwt1gt_2),243).
orientation(mwt1gt_1,160,244). coords(mwt1gt_1,65,184,244).  visible(mwt1gt_1,244). happensAt(walking(mwt1gt_1),244). orientation(mwt1gt_2,110,244). coords(mwt1gt_2,256,22,244).  visible(mwt1gt_2,244). happensAt(walking(mwt1gt_2),244).
orientation(mwt1gt_1,160,245). coords(mwt1gt_1,65,184,245).  visible(mwt1gt_1,245). happensAt(walking(mwt1gt_1),245). orientation(mwt1gt_2,110,245). coords(mwt1gt_2,251,22,245).  visible(mwt1gt_2,245). happensAt(walking(mwt1gt_2),245).
orientation(mwt1gt_1,160,246). coords(mwt1gt_1,66,184,246).  visible(mwt1gt_1,246). happensAt(walking(mwt1gt_1),246). orientation(mwt1gt_2,110,246). coords(mwt1gt_2,248,22,246).  visible(mwt1gt_2,246). happensAt(walking(mwt1gt_2),246).
orientation(mwt1gt_1,160,247). coords(mwt1gt_1,66,184,247).  visible(mwt1gt_1,247). happensAt(walking(mwt1gt_1),247). orientation(mwt1gt_2,110,247). coords(mwt1gt_2,247,22,247).  visible(mwt1gt_2,247). happensAt(walking(mwt1gt_2),247).
orientation(mwt1gt_1,160,248). coords(mwt1gt_1,67,183,248).  visible(mwt1gt_1,248). happensAt(walking(mwt1gt_1),248). orientation(mwt1gt_2,110,248). coords(mwt1gt_2,247,22,248).  visible(mwt1gt_2,248). happensAt(walking(mwt1gt_2),248).
orientation(mwt1gt_1,160,249). coords(mwt1gt_1,68,183,249).  visible(mwt1gt_1,249). happensAt(walking(mwt1gt_1),249). orientation(mwt1gt_2,110,249). coords(mwt1gt_2,245,24,249).  visible(mwt1gt_2,249). happensAt(walking(mwt1gt_2),249).
orientation(mwt1gt_1,160,250). coords(mwt1gt_1,69,183,250).  visible(mwt1gt_1,250). happensAt(walking(mwt1gt_1),250). orientation(mwt1gt_2,110,250). coords(mwt1gt_2,244,27,250).  visible(mwt1gt_2,250). happensAt(walking(mwt1gt_2),250).
orientation(mwt1gt_1,160,251). coords(mwt1gt_1,70,183,251).  visible(mwt1gt_1,251). happensAt(walking(mwt1gt_1),251). orientation(mwt1gt_2,110,251). coords(mwt1gt_2,243,29,251).  visible(mwt1gt_2,251). happensAt(walking(mwt1gt_2),251).
orientation(mwt1gt_1,160,252). coords(mwt1gt_1,71,183,252).  visible(mwt1gt_1,252). happensAt(walking(mwt1gt_1),252). orientation(mwt1gt_2,110,252). coords(mwt1gt_2,242,30,252).  visible(mwt1gt_2,252). happensAt(walking(mwt1gt_2),252).
orientation(mwt1gt_1,160,253). coords(mwt1gt_1,72,181,253).  visible(mwt1gt_1,253). happensAt(walking(mwt1gt_1),253). orientation(mwt1gt_2,110,253). coords(mwt1gt_2,242,31,253).  visible(mwt1gt_2,253). happensAt(walking(mwt1gt_2),253).
orientation(mwt1gt_1,160,254). coords(mwt1gt_1,72,181,254).  visible(mwt1gt_1,254). happensAt(walking(mwt1gt_1),254). orientation(mwt1gt_2,110,254). coords(mwt1gt_2,241,33,254).  visible(mwt1gt_2,254). happensAt(walking(mwt1gt_2),254).
orientation(mwt1gt_1,160,255). coords(mwt1gt_1,72,181,255).  visible(mwt1gt_1,255). happensAt(walking(mwt1gt_1),255). orientation(mwt1gt_2,110,255). coords(mwt1gt_2,241,33,255).  visible(mwt1gt_2,255). happensAt(walking(mwt1gt_2),255).
orientation(mwt1gt_1,160,256). coords(mwt1gt_1,73,180,256).  visible(mwt1gt_1,256). happensAt(walking(mwt1gt_1),256). orientation(mwt1gt_2,130,256). coords(mwt1gt_2,237,35,256).  visible(mwt1gt_2,256). happensAt(walking(mwt1gt_2),256).
orientation(mwt1gt_1,160,257). coords(mwt1gt_1,73,180,257).  visible(mwt1gt_1,257). happensAt(walking(mwt1gt_1),257). orientation(mwt1gt_2,130,257). coords(mwt1gt_2,235,35,257).  visible(mwt1gt_2,257). happensAt(walking(mwt1gt_2),257).
#show holdsAt/2.