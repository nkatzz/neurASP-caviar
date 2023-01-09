
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

	orientation(mws1gt_1,155,226). coords(mws1gt_1,125,130,226).  visible(mws1gt_1,226). happensAt(walking(mws1gt_1),226). orientation(mws1gt_5,161,226). coords(mws1gt_5,45,120,226).  visible(mws1gt_5,226). happensAt(walking(mws1gt_5),226).
orientation(mws1gt_1,155,227). coords(mws1gt_1,124,130,227).  visible(mws1gt_1,227). happensAt(walking(mws1gt_1),227). orientation(mws1gt_5,161,227). coords(mws1gt_5,45,121,227).  visible(mws1gt_5,227). happensAt(walking(mws1gt_5),227).
orientation(mws1gt_1,155,228). coords(mws1gt_1,123,129,228).  visible(mws1gt_1,228). happensAt(walking(mws1gt_1),228). orientation(mws1gt_5,161,228). coords(mws1gt_5,45,124,228).  visible(mws1gt_5,228). happensAt(walking(mws1gt_5),228).
orientation(mws1gt_1,155,229). coords(mws1gt_1,122,129,229).  visible(mws1gt_1,229). happensAt(walking(mws1gt_1),229). orientation(mws1gt_5,161,229). coords(mws1gt_5,45,126,229).  visible(mws1gt_5,229). happensAt(walking(mws1gt_5),229).
orientation(mws1gt_1,155,230). coords(mws1gt_1,121,128,230).  visible(mws1gt_1,230). happensAt(walking(mws1gt_1),230). orientation(mws1gt_5,161,230). coords(mws1gt_5,45,127,230).  visible(mws1gt_5,230). happensAt(walking(mws1gt_5),230).
orientation(mws1gt_1,155,231). coords(mws1gt_1,120,127,231).  visible(mws1gt_1,231). happensAt(walking(mws1gt_1),231). orientation(mws1gt_5,161,231). coords(mws1gt_5,45,128,231).  visible(mws1gt_5,231). happensAt(walking(mws1gt_5),231).
orientation(mws1gt_1,155,232). coords(mws1gt_1,118,127,232).  visible(mws1gt_1,232). happensAt(walking(mws1gt_1),232). orientation(mws1gt_5,161,232). coords(mws1gt_5,45,128,232).  visible(mws1gt_5,232). happensAt(walking(mws1gt_5),232).
orientation(mws1gt_1,155,233). coords(mws1gt_1,118,126,233).  visible(mws1gt_1,233). happensAt(walking(mws1gt_1),233). orientation(mws1gt_5,161,233). coords(mws1gt_5,45,129,233).  visible(mws1gt_5,233). happensAt(walking(mws1gt_5),233).
orientation(mws1gt_1,155,234). coords(mws1gt_1,117,125,234).  visible(mws1gt_1,234). happensAt(walking(mws1gt_1),234). orientation(mws1gt_5,161,234). coords(mws1gt_5,45,129,234).  visible(mws1gt_5,234). happensAt(walking(mws1gt_5),234).
orientation(mws1gt_1,155,235). coords(mws1gt_1,117,125,235).  visible(mws1gt_1,235). happensAt(walking(mws1gt_1),235). orientation(mws1gt_5,161,235). coords(mws1gt_5,45,130,235).  visible(mws1gt_5,235). happensAt(walking(mws1gt_5),235).
orientation(mws1gt_1,155,236). coords(mws1gt_1,117,125,236).  visible(mws1gt_1,236). happensAt(walking(mws1gt_1),236). orientation(mws1gt_5,161,236). coords(mws1gt_5,44,131,236).  visible(mws1gt_5,236). happensAt(walking(mws1gt_5),236).
orientation(mws1gt_1,155,237). coords(mws1gt_1,117,124,237).  visible(mws1gt_1,237). happensAt(walking(mws1gt_1),237). orientation(mws1gt_5,161,237). coords(mws1gt_5,43,132,237).  visible(mws1gt_5,237). happensAt(walking(mws1gt_5),237).
orientation(mws1gt_1,155,238). coords(mws1gt_1,116,123,238).  visible(mws1gt_1,238). happensAt(walking(mws1gt_1),238). orientation(mws1gt_5,161,238). coords(mws1gt_5,43,133,238).  visible(mws1gt_5,238). happensAt(walking(mws1gt_5),238).
orientation(mws1gt_1,155,239). coords(mws1gt_1,116,123,239).  visible(mws1gt_1,239). happensAt(walking(mws1gt_1),239). orientation(mws1gt_5,161,239). coords(mws1gt_5,43,134,239).  visible(mws1gt_5,239). happensAt(walking(mws1gt_5),239).
orientation(mws1gt_1,155,240). coords(mws1gt_1,116,123,240).  visible(mws1gt_1,240). happensAt(walking(mws1gt_1),240). orientation(mws1gt_5,161,240). coords(mws1gt_5,43,134,240).  visible(mws1gt_5,240). happensAt(walking(mws1gt_5),240).
orientation(mws1gt_1,155,241). coords(mws1gt_1,116,123,241).  visible(mws1gt_1,241). happensAt(walking(mws1gt_1),241). orientation(mws1gt_5,161,241). coords(mws1gt_5,43,134,241).  visible(mws1gt_5,241). happensAt(walking(mws1gt_5),241).
orientation(mws1gt_1,155,242). coords(mws1gt_1,115,122,242).  visible(mws1gt_1,242). happensAt(walking(mws1gt_1),242). orientation(mws1gt_5,161,242). coords(mws1gt_5,42,135,242).  visible(mws1gt_5,242). happensAt(walking(mws1gt_5),242).
orientation(mws1gt_1,155,243). coords(mws1gt_1,114,122,243).  visible(mws1gt_1,243). happensAt(walking(mws1gt_1),243). orientation(mws1gt_5,161,243). coords(mws1gt_5,42,135,243).  visible(mws1gt_5,243). happensAt(walking(mws1gt_5),243).
orientation(mws1gt_1,155,244). coords(mws1gt_1,113,122,244).  visible(mws1gt_1,244). happensAt(walking(mws1gt_1),244). orientation(mws1gt_5,161,244). coords(mws1gt_5,42,135,244).  visible(mws1gt_5,244). happensAt(walking(mws1gt_5),244).
orientation(mws1gt_1,155,245). coords(mws1gt_1,112,122,245).  visible(mws1gt_1,245). happensAt(walking(mws1gt_1),245). orientation(mws1gt_5,161,245). coords(mws1gt_5,42,135,245).  visible(mws1gt_5,245). happensAt(walking(mws1gt_5),245).
orientation(mws1gt_1,155,246). coords(mws1gt_1,110,122,246).  visible(mws1gt_1,246). happensAt(walking(mws1gt_1),246). orientation(mws1gt_5,161,246). coords(mws1gt_5,41,135,246).  visible(mws1gt_5,246). happensAt(walking(mws1gt_5),246).
orientation(mws1gt_1,155,247). coords(mws1gt_1,109,121,247).  visible(mws1gt_1,247). happensAt(walking(mws1gt_1),247). orientation(mws1gt_5,161,247). coords(mws1gt_5,41,137,247).  visible(mws1gt_5,247). happensAt(walking(mws1gt_5),247).
orientation(mws1gt_1,155,248). coords(mws1gt_1,109,121,248).  visible(mws1gt_1,248). happensAt(walking(mws1gt_1),248). orientation(mws1gt_5,161,248). coords(mws1gt_5,41,138,248).  visible(mws1gt_5,248). happensAt(walking(mws1gt_5),248).
orientation(mws1gt_1,155,249). coords(mws1gt_1,108,121,249).  visible(mws1gt_1,249). happensAt(walking(mws1gt_1),249). orientation(mws1gt_5,161,249). coords(mws1gt_5,41,138,249).  visible(mws1gt_5,249). happensAt(walking(mws1gt_5),249).
orientation(mws1gt_1,155,250). coords(mws1gt_1,106,120,250).  visible(mws1gt_1,250). happensAt(walking(mws1gt_1),250). orientation(mws1gt_5,161,250). coords(mws1gt_5,41,139,250).  visible(mws1gt_5,250). happensAt(walking(mws1gt_5),250).
#show holdsAt/2.