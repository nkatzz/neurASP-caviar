
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

	orientation(mwt2gt_1,142,229). coords(mwt2gt_1,143,60,229).  visible(mwt2gt_1,229). happensAt(walking(mwt2gt_1),229). orientation(mwt2gt_2,146,229). coords(mwt2gt_2,220,184,229).  visible(mwt2gt_2,229). happensAt(walking(mwt2gt_2),229).
orientation(mwt2gt_1,142,230). coords(mwt2gt_1,142,61,230).  visible(mwt2gt_1,230). happensAt(walking(mwt2gt_1),230). orientation(mwt2gt_2,146,230). coords(mwt2gt_2,218,180,230).  visible(mwt2gt_2,230). happensAt(walking(mwt2gt_2),230).
orientation(mwt2gt_1,142,231). coords(mwt2gt_1,142,61,231).  visible(mwt2gt_1,231). happensAt(walking(mwt2gt_1),231). orientation(mwt2gt_2,146,231). coords(mwt2gt_2,218,180,231).  visible(mwt2gt_2,231). happensAt(walking(mwt2gt_2),231).
orientation(mwt2gt_1,142,232). coords(mwt2gt_1,142,61,232).  visible(mwt2gt_1,232). happensAt(walking(mwt2gt_1),232). orientation(mwt2gt_2,146,232). coords(mwt2gt_2,217,174,232).  visible(mwt2gt_2,232). happensAt(walking(mwt2gt_2),232).
orientation(mwt2gt_1,142,233). coords(mwt2gt_1,142,61,233).  visible(mwt2gt_1,233). happensAt(walking(mwt2gt_1),233). orientation(mwt2gt_2,146,233). coords(mwt2gt_2,216,174,233).  visible(mwt2gt_2,233). happensAt(walking(mwt2gt_2),233).
orientation(mwt2gt_1,142,234). coords(mwt2gt_1,141,62,234).  visible(mwt2gt_1,234). happensAt(walking(mwt2gt_1),234). orientation(mwt2gt_2,146,234). coords(mwt2gt_2,216,172,234).  visible(mwt2gt_2,234). happensAt(walking(mwt2gt_2),234).
orientation(mwt2gt_1,142,235). coords(mwt2gt_1,140,64,235).  visible(mwt2gt_1,235). happensAt(walking(mwt2gt_1),235). orientation(mwt2gt_2,146,235). coords(mwt2gt_2,215,171,235).  visible(mwt2gt_2,235). happensAt(walking(mwt2gt_2),235).
orientation(mwt2gt_1,142,236). coords(mwt2gt_1,140,65,236).  visible(mwt2gt_1,236). happensAt(walking(mwt2gt_1),236). orientation(mwt2gt_2,146,236). coords(mwt2gt_2,215,170,236).  visible(mwt2gt_2,236). happensAt(walking(mwt2gt_2),236).
orientation(mwt2gt_1,142,237). coords(mwt2gt_1,140,65,237).  visible(mwt2gt_1,237). happensAt(walking(mwt2gt_1),237). orientation(mwt2gt_2,146,237). coords(mwt2gt_2,214,169,237).  visible(mwt2gt_2,237). happensAt(walking(mwt2gt_2),237).
orientation(mwt2gt_1,142,238). coords(mwt2gt_1,139,66,238).  visible(mwt2gt_1,238). happensAt(walking(mwt2gt_1),238). orientation(mwt2gt_2,146,238). coords(mwt2gt_2,213,168,238).  visible(mwt2gt_2,238). happensAt(walking(mwt2gt_2),238).
orientation(mwt2gt_1,142,239). coords(mwt2gt_1,140,67,239).  visible(mwt2gt_1,239). happensAt(walking(mwt2gt_1),239). orientation(mwt2gt_2,146,239). coords(mwt2gt_2,213,167,239).  visible(mwt2gt_2,239). happensAt(walking(mwt2gt_2),239).
orientation(mwt2gt_1,138,240). coords(mwt2gt_1,140,67,240).  visible(mwt2gt_1,240). happensAt(walking(mwt2gt_1),240). orientation(mwt2gt_2,146,240). coords(mwt2gt_2,211,165,240).  visible(mwt2gt_2,240). happensAt(walking(mwt2gt_2),240).
orientation(mwt2gt_1,138,241). coords(mwt2gt_1,140,68,241).  visible(mwt2gt_1,241). happensAt(walking(mwt2gt_1),241). orientation(mwt2gt_2,146,241). coords(mwt2gt_2,210,163,241).  visible(mwt2gt_2,241). happensAt(walking(mwt2gt_2),241).
orientation(mwt2gt_1,138,242). coords(mwt2gt_1,140,68,242).  visible(mwt2gt_1,242). happensAt(walking(mwt2gt_1),242). orientation(mwt2gt_2,146,242). coords(mwt2gt_2,207,160,242).  visible(mwt2gt_2,242). happensAt(walking(mwt2gt_2),242).
orientation(mwt2gt_1,138,243). coords(mwt2gt_1,140,68,243).  visible(mwt2gt_1,243). happensAt(walking(mwt2gt_1),243). orientation(mwt2gt_2,146,243). coords(mwt2gt_2,206,157,243).  visible(mwt2gt_2,243). happensAt(walking(mwt2gt_2),243).
orientation(mwt2gt_1,138,244). coords(mwt2gt_1,140,68,244).  visible(mwt2gt_1,244). happensAt(walking(mwt2gt_1),244). orientation(mwt2gt_2,148,244). coords(mwt2gt_2,204,154,244).  visible(mwt2gt_2,244). happensAt(walking(mwt2gt_2),244).
orientation(mwt2gt_1,138,245). coords(mwt2gt_1,140,69,245).  visible(mwt2gt_1,245). happensAt(walking(mwt2gt_1),245). orientation(mwt2gt_2,148,245). coords(mwt2gt_2,202,154,245).  visible(mwt2gt_2,245). happensAt(walking(mwt2gt_2),245).
orientation(mwt2gt_1,138,246). coords(mwt2gt_1,140,68,246).  visible(mwt2gt_1,246). happensAt(walking(mwt2gt_1),246). orientation(mwt2gt_2,148,246). coords(mwt2gt_2,201,153,246).  visible(mwt2gt_2,246). happensAt(walking(mwt2gt_2),246).
orientation(mwt2gt_1,138,247). coords(mwt2gt_1,140,68,247).  visible(mwt2gt_1,247). happensAt(walking(mwt2gt_1),247). orientation(mwt2gt_2,148,247). coords(mwt2gt_2,201,153,247).  visible(mwt2gt_2,247). happensAt(walking(mwt2gt_2),247).
orientation(mwt2gt_1,138,248). coords(mwt2gt_1,139,69,248).  visible(mwt2gt_1,248). happensAt(walking(mwt2gt_1),248). orientation(mwt2gt_2,148,248). coords(mwt2gt_2,200,151,248).  visible(mwt2gt_2,248). happensAt(walking(mwt2gt_2),248).
orientation(mwt2gt_1,138,249). coords(mwt2gt_1,139,69,249).  visible(mwt2gt_1,249). happensAt(walking(mwt2gt_1),249). orientation(mwt2gt_2,148,249). coords(mwt2gt_2,200,150,249).  visible(mwt2gt_2,249). happensAt(walking(mwt2gt_2),249).
orientation(mwt2gt_1,138,250). coords(mwt2gt_1,139,69,250).  visible(mwt2gt_1,250). happensAt(walking(mwt2gt_1),250). orientation(mwt2gt_2,145,250). coords(mwt2gt_2,199,150,250).  visible(mwt2gt_2,250). happensAt(walking(mwt2gt_2),250).
orientation(mwt2gt_1,138,251). coords(mwt2gt_1,139,69,251).  visible(mwt2gt_1,251). happensAt(walking(mwt2gt_1),251). orientation(mwt2gt_2,145,251). coords(mwt2gt_2,199,148,251).  visible(mwt2gt_2,251). happensAt(walking(mwt2gt_2),251).
orientation(mwt2gt_1,138,252). coords(mwt2gt_1,140,70,252).  visible(mwt2gt_1,252). happensAt(walking(mwt2gt_1),252). orientation(mwt2gt_2,145,252). coords(mwt2gt_2,198,147,252).  visible(mwt2gt_2,252). happensAt(walking(mwt2gt_2),252).
orientation(mwt2gt_1,138,253). coords(mwt2gt_1,139,71,253).  visible(mwt2gt_1,253). happensAt(walking(mwt2gt_1),253). orientation(mwt2gt_2,145,253). coords(mwt2gt_2,196,145,253).  visible(mwt2gt_2,253). happensAt(walking(mwt2gt_2),253).
#show holdsAt/2.