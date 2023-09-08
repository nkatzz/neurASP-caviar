
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

	orientation(fra2gt_3,161,226). coords(fra2gt_3,51,161,226).  visible(fra2gt_3,226). happensAt(walking(fra2gt_3),226). orientation(fra2gt_4,163,226). coords(fra2gt_4,197,209,226).  visible(fra2gt_4,226). happensAt(walking(fra2gt_4),226).
orientation(fra2gt_3,161,227). coords(fra2gt_3,51,160,227).  visible(fra2gt_3,227). happensAt(walking(fra2gt_3),227). orientation(fra2gt_4,165,227). coords(fra2gt_4,195,205,227).  visible(fra2gt_4,227). happensAt(walking(fra2gt_4),227).
orientation(fra2gt_3,161,228). coords(fra2gt_3,51,159,228).  visible(fra2gt_3,228). happensAt(walking(fra2gt_3),228). orientation(fra2gt_4,165,228). coords(fra2gt_4,195,203,228).  visible(fra2gt_4,228). happensAt(walking(fra2gt_4),228).
orientation(fra2gt_3,161,229). coords(fra2gt_3,51,159,229).  visible(fra2gt_3,229). happensAt(walking(fra2gt_3),229). orientation(fra2gt_4,159,229). coords(fra2gt_4,194,201,229).  visible(fra2gt_4,229). happensAt(walking(fra2gt_4),229).
orientation(fra2gt_3,161,230). coords(fra2gt_3,53,159,230).  visible(fra2gt_3,230). happensAt(walking(fra2gt_3),230). orientation(fra2gt_4,159,230). coords(fra2gt_4,194,199,230).  visible(fra2gt_4,230). happensAt(walking(fra2gt_4),230).
orientation(fra2gt_3,161,231). coords(fra2gt_3,54,158,231).  visible(fra2gt_3,231). happensAt(walking(fra2gt_3),231). orientation(fra2gt_4,159,231). coords(fra2gt_4,193,197,231).  visible(fra2gt_4,231). happensAt(walking(fra2gt_4),231).
orientation(fra2gt_3,161,232). coords(fra2gt_3,54,158,232).  visible(fra2gt_3,232). happensAt(walking(fra2gt_3),232). orientation(fra2gt_4,159,232). coords(fra2gt_4,192,196,232).  visible(fra2gt_4,232). happensAt(walking(fra2gt_4),232).
orientation(fra2gt_3,161,233). coords(fra2gt_3,55,158,233).  visible(fra2gt_3,233). happensAt(walking(fra2gt_3),233). orientation(fra2gt_4,159,233). coords(fra2gt_4,190,194,233).  visible(fra2gt_4,233). happensAt(walking(fra2gt_4),233).
orientation(fra2gt_3,161,234). coords(fra2gt_3,55,157,234).  visible(fra2gt_3,234). happensAt(walking(fra2gt_3),234). orientation(fra2gt_4,159,234). coords(fra2gt_4,189,191,234).  visible(fra2gt_4,234). happensAt(walking(fra2gt_4),234).
orientation(fra2gt_3,161,235). coords(fra2gt_3,55,156,235).  visible(fra2gt_3,235). happensAt(walking(fra2gt_3),235). orientation(fra2gt_4,159,235). coords(fra2gt_4,188,189,235).  visible(fra2gt_4,235). happensAt(walking(fra2gt_4),235).
orientation(fra2gt_3,161,236). coords(fra2gt_3,55,156,236).  visible(fra2gt_3,236). happensAt(walking(fra2gt_3),236). orientation(fra2gt_4,159,236). coords(fra2gt_4,184,187,236).  visible(fra2gt_4,236). happensAt(walking(fra2gt_4),236).
orientation(fra2gt_3,156,237). coords(fra2gt_3,55,155,237).  visible(fra2gt_3,237). happensAt(walking(fra2gt_3),237). orientation(fra2gt_4,159,237). coords(fra2gt_4,182,185,237).  visible(fra2gt_4,237). happensAt(walking(fra2gt_4),237).
orientation(fra2gt_3,151,238). coords(fra2gt_3,55,153,238).  visible(fra2gt_3,238). happensAt(walking(fra2gt_3),238). orientation(fra2gt_4,159,238). coords(fra2gt_4,179,183,238).  visible(fra2gt_4,238). happensAt(walking(fra2gt_4),238).
orientation(fra2gt_3,151,239). coords(fra2gt_3,55,152,239).  visible(fra2gt_3,239). happensAt(walking(fra2gt_3),239). orientation(fra2gt_4,159,239). coords(fra2gt_4,178,182,239).  visible(fra2gt_4,239). happensAt(walking(fra2gt_4),239).
orientation(fra2gt_3,151,240). coords(fra2gt_3,56,151,240).  visible(fra2gt_3,240). happensAt(walking(fra2gt_3),240). orientation(fra2gt_4,159,240). coords(fra2gt_4,177,181,240).  visible(fra2gt_4,240). happensAt(walking(fra2gt_4),240).
orientation(fra2gt_3,151,241). coords(fra2gt_3,56,149,241).  visible(fra2gt_3,241). happensAt(walking(fra2gt_3),241). orientation(fra2gt_4,159,241). coords(fra2gt_4,177,180,241).  visible(fra2gt_4,241). happensAt(walking(fra2gt_4),241).
orientation(fra2gt_3,151,242). coords(fra2gt_3,57,149,242).  visible(fra2gt_3,242). happensAt(walking(fra2gt_3),242). orientation(fra2gt_4,159,242). coords(fra2gt_4,176,179,242).  visible(fra2gt_4,242). happensAt(walking(fra2gt_4),242).
orientation(fra2gt_3,151,243). coords(fra2gt_3,58,148,243).  visible(fra2gt_3,243). happensAt(walking(fra2gt_3),243). orientation(fra2gt_4,155,243). coords(fra2gt_4,175,178,243).  visible(fra2gt_4,243). happensAt(walking(fra2gt_4),243).
orientation(fra2gt_3,151,244). coords(fra2gt_3,60,148,244).  visible(fra2gt_3,244). happensAt(walking(fra2gt_3),244). orientation(fra2gt_4,155,244). coords(fra2gt_4,174,176,244).  visible(fra2gt_4,244). happensAt(walking(fra2gt_4),244).
orientation(fra2gt_3,151,245). coords(fra2gt_3,60,148,245).  visible(fra2gt_3,245). happensAt(walking(fra2gt_3),245). orientation(fra2gt_4,155,245). coords(fra2gt_4,173,174,245).  visible(fra2gt_4,245). happensAt(walking(fra2gt_4),245).
orientation(fra2gt_3,151,246). coords(fra2gt_3,60,148,246).  visible(fra2gt_3,246). happensAt(walking(fra2gt_3),246). orientation(fra2gt_4,155,246). coords(fra2gt_4,171,172,246).  visible(fra2gt_4,246). happensAt(walking(fra2gt_4),246).
orientation(fra2gt_3,151,247). coords(fra2gt_3,60,147,247).  visible(fra2gt_3,247). happensAt(walking(fra2gt_3),247). orientation(fra2gt_4,155,247). coords(fra2gt_4,168,169,247).  visible(fra2gt_4,247). happensAt(walking(fra2gt_4),247).
orientation(fra2gt_3,151,248). coords(fra2gt_3,61,146,248).  visible(fra2gt_3,248). happensAt(walking(fra2gt_3),248). orientation(fra2gt_4,155,248). coords(fra2gt_4,166,167,248).  visible(fra2gt_4,248). happensAt(walking(fra2gt_4),248).
orientation(fra2gt_3,151,249). coords(fra2gt_3,61,145,249).  visible(fra2gt_3,249). happensAt(walking(fra2gt_3),249). orientation(fra2gt_4,155,249). coords(fra2gt_4,164,163,249).  visible(fra2gt_4,249). happensAt(walking(fra2gt_4),249).
orientation(fra2gt_3,151,250). coords(fra2gt_3,62,144,250).  visible(fra2gt_3,250). happensAt(walking(fra2gt_3),250). orientation(fra2gt_4,155,250). coords(fra2gt_4,163,160,250).  visible(fra2gt_4,250). happensAt(walking(fra2gt_4),250).
#show holdsAt/2.