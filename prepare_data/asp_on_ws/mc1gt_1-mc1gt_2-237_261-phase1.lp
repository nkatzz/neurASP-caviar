
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

	orientation(mc1gt_1,142,237). coords(mc1gt_1,189,140,237).  visible(mc1gt_1,237). happensAt(walking(mc1gt_1),237). orientation(mc1gt_2,146,237). coords(mc1gt_2,132,135,237).  visible(mc1gt_2,237). happensAt(walking(mc1gt_2),237).
orientation(mc1gt_1,142,238). coords(mc1gt_1,190,141,238).  visible(mc1gt_1,238). happensAt(walking(mc1gt_1),238). orientation(mc1gt_2,146,238). coords(mc1gt_2,132,136,238).  visible(mc1gt_2,238). happensAt(walking(mc1gt_2),238).
orientation(mc1gt_1,142,239). coords(mc1gt_1,190,141,239).  visible(mc1gt_1,239). happensAt(walking(mc1gt_1),239). orientation(mc1gt_2,146,239). coords(mc1gt_2,132,136,239).  visible(mc1gt_2,239). happensAt(walking(mc1gt_2),239).
orientation(mc1gt_1,145,240). coords(mc1gt_1,190,142,240).  visible(mc1gt_1,240). happensAt(walking(mc1gt_1),240). orientation(mc1gt_2,146,240). coords(mc1gt_2,133,137,240).  visible(mc1gt_2,240). happensAt(walking(mc1gt_2),240).
orientation(mc1gt_1,147,241). coords(mc1gt_1,191,142,241).  visible(mc1gt_1,241). happensAt(walking(mc1gt_1),241). orientation(mc1gt_2,146,241). coords(mc1gt_2,133,137,241).  visible(mc1gt_2,241). happensAt(walking(mc1gt_2),241).
orientation(mc1gt_1,146,242). coords(mc1gt_1,192,143,242).  visible(mc1gt_1,242). happensAt(walking(mc1gt_1),242). orientation(mc1gt_2,146,242). coords(mc1gt_2,133,137,242).  visible(mc1gt_2,242). happensAt(walking(mc1gt_2),242).
orientation(mc1gt_1,148,243). coords(mc1gt_1,194,144,243).  visible(mc1gt_1,243). happensAt(walking(mc1gt_1),243). orientation(mc1gt_2,146,243). coords(mc1gt_2,134,137,243).  visible(mc1gt_2,243). happensAt(walking(mc1gt_2),243).
orientation(mc1gt_1,145,244). coords(mc1gt_1,197,145,244).  visible(mc1gt_1,244). happensAt(walking(mc1gt_1),244). orientation(mc1gt_2,146,244). coords(mc1gt_2,134,138,244).  visible(mc1gt_2,244). happensAt(walking(mc1gt_2),244).
orientation(mc1gt_1,142,245). coords(mc1gt_1,198,149,245).  visible(mc1gt_1,245). happensAt(walking(mc1gt_1),245). orientation(mc1gt_2,146,245). coords(mc1gt_2,135,138,245).  visible(mc1gt_2,245). happensAt(walking(mc1gt_2),245).
orientation(mc1gt_1,142,246). coords(mc1gt_1,200,151,246).  visible(mc1gt_1,246). happensAt(walking(mc1gt_1),246). orientation(mc1gt_2,146,246). coords(mc1gt_2,135,139,246).  visible(mc1gt_2,246). happensAt(walking(mc1gt_2),246).
orientation(mc1gt_1,142,247). coords(mc1gt_1,201,154,247).  visible(mc1gt_1,247). happensAt(walking(mc1gt_1),247). orientation(mc1gt_2,146,247). coords(mc1gt_2,136,139,247).  visible(mc1gt_2,247). happensAt(walking(mc1gt_2),247).
orientation(mc1gt_1,142,248). coords(mc1gt_1,202,155,248).  visible(mc1gt_1,248). happensAt(walking(mc1gt_1),248). orientation(mc1gt_2,146,248). coords(mc1gt_2,137,140,248).  visible(mc1gt_2,248). happensAt(walking(mc1gt_2),248).
orientation(mc1gt_1,142,249). coords(mc1gt_1,203,157,249).  visible(mc1gt_1,249). happensAt(walking(mc1gt_1),249). orientation(mc1gt_2,146,249). coords(mc1gt_2,137,140,249).  visible(mc1gt_2,249). happensAt(walking(mc1gt_2),249).
orientation(mc1gt_1,142,250). coords(mc1gt_1,204,158,250).  visible(mc1gt_1,250). happensAt(walking(mc1gt_1),250). orientation(mc1gt_2,146,250). coords(mc1gt_2,138,142,250).  visible(mc1gt_2,250). happensAt(walking(mc1gt_2),250).
orientation(mc1gt_1,142,251). coords(mc1gt_1,205,159,251).  visible(mc1gt_1,251). happensAt(walking(mc1gt_1),251). orientation(mc1gt_2,146,251). coords(mc1gt_2,138,142,251).  visible(mc1gt_2,251). happensAt(walking(mc1gt_2),251).
orientation(mc1gt_1,144,252). coords(mc1gt_1,205,159,252).  visible(mc1gt_1,252). happensAt(walking(mc1gt_1),252). orientation(mc1gt_2,146,252). coords(mc1gt_2,139,142,252).  visible(mc1gt_2,252). happensAt(walking(mc1gt_2),252).
orientation(mc1gt_1,145,253). coords(mc1gt_1,206,160,253).  visible(mc1gt_1,253). happensAt(walking(mc1gt_1),253). orientation(mc1gt_2,146,253). coords(mc1gt_2,141,142,253).  visible(mc1gt_2,253). happensAt(walking(mc1gt_2),253).
orientation(mc1gt_1,146,254). coords(mc1gt_1,206,161,254).  visible(mc1gt_1,254). happensAt(walking(mc1gt_1),254). orientation(mc1gt_2,146,254). coords(mc1gt_2,142,144,254).  visible(mc1gt_2,254). happensAt(walking(mc1gt_2),254).
orientation(mc1gt_1,147,255). coords(mc1gt_1,206,162,255).  visible(mc1gt_1,255). happensAt(walking(mc1gt_1),255). orientation(mc1gt_2,146,255). coords(mc1gt_2,143,143,255).  visible(mc1gt_2,255). happensAt(walking(mc1gt_2),255).
orientation(mc1gt_1,146,256). coords(mc1gt_1,206,163,256).  visible(mc1gt_1,256). happensAt(walking(mc1gt_1),256). orientation(mc1gt_2,146,256). coords(mc1gt_2,144,144,256).  visible(mc1gt_2,256). happensAt(walking(mc1gt_2),256).
orientation(mc1gt_1,147,257). coords(mc1gt_1,206,163,257).  visible(mc1gt_1,257). happensAt(walking(mc1gt_1),257). orientation(mc1gt_2,146,257). coords(mc1gt_2,145,145,257).  visible(mc1gt_2,257). happensAt(walking(mc1gt_2),257).
orientation(mc1gt_1,153,258). coords(mc1gt_1,207,166,258).  visible(mc1gt_1,258). happensAt(walking(mc1gt_1),258). orientation(mc1gt_2,146,258). coords(mc1gt_2,148,145,258).  visible(mc1gt_2,258). happensAt(walking(mc1gt_2),258).
orientation(mc1gt_1,155,259). coords(mc1gt_1,207,170,259).  visible(mc1gt_1,259). happensAt(walking(mc1gt_1),259). orientation(mc1gt_2,146,259). coords(mc1gt_2,149,146,259).  visible(mc1gt_2,259). happensAt(walking(mc1gt_2),259).
orientation(mc1gt_1,159,260). coords(mc1gt_1,207,174,260).  visible(mc1gt_1,260). happensAt(walking(mc1gt_1),260). orientation(mc1gt_2,146,260). coords(mc1gt_2,149,146,260).  visible(mc1gt_2,260). happensAt(walking(mc1gt_2),260).
orientation(mc1gt_1,164,261). coords(mc1gt_1,207,178,261).  visible(mc1gt_1,261). happensAt(walking(mc1gt_1),261). orientation(mc1gt_2,146,261). coords(mc1gt_2,150,147,261).  visible(mc1gt_2,261). happensAt(walking(mc1gt_2),261).
#show holdsAt/2.