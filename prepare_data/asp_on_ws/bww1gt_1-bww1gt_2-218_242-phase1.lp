
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

	orientation(bww1gt_1,175,218). coords(bww1gt_1,99,264,218).  visible(bww1gt_1,218). happensAt(walking(bww1gt_1),218). orientation(bww1gt_2,167,218). coords(bww1gt_2,35,165,218).  visible(bww1gt_2,218). happensAt(walking(bww1gt_2),218).
orientation(bww1gt_1,176,219). coords(bww1gt_1,98,264,219).  visible(bww1gt_1,219). happensAt(walking(bww1gt_1),219). orientation(bww1gt_2,167,219). coords(bww1gt_2,35,165,219).  visible(bww1gt_2,219). happensAt(walking(bww1gt_2),219).
orientation(bww1gt_1,176,220). coords(bww1gt_1,98,263,220).  visible(bww1gt_1,220). happensAt(walking(bww1gt_1),220). orientation(bww1gt_2,167,220). coords(bww1gt_2,35,165,220).  visible(bww1gt_2,220). happensAt(walking(bww1gt_2),220).
orientation(bww1gt_1,174,221). coords(bww1gt_1,97,263,221).  visible(bww1gt_1,221). happensAt(walking(bww1gt_1),221). orientation(bww1gt_2,167,221). coords(bww1gt_2,35,166,221).  visible(bww1gt_2,221). happensAt(walking(bww1gt_2),221).
orientation(bww1gt_1,173,222). coords(bww1gt_1,96,263,222).  visible(bww1gt_1,222). happensAt(walking(bww1gt_1),222). orientation(bww1gt_2,167,222). coords(bww1gt_2,35,166,222).  visible(bww1gt_2,222). happensAt(walking(bww1gt_2),222).
orientation(bww1gt_1,169,223). coords(bww1gt_1,94,263,223).  visible(bww1gt_1,223). happensAt(walking(bww1gt_1),223). orientation(bww1gt_2,167,223). coords(bww1gt_2,35,166,223).  visible(bww1gt_2,223). happensAt(walking(bww1gt_2),223).
orientation(bww1gt_1,173,224). coords(bww1gt_1,94,263,224).  visible(bww1gt_1,224). happensAt(walking(bww1gt_1),224). orientation(bww1gt_2,167,224). coords(bww1gt_2,35,166,224).  visible(bww1gt_2,224). happensAt(walking(bww1gt_2),224).
orientation(bww1gt_1,172,225). coords(bww1gt_1,94,262,225).  visible(bww1gt_1,225). happensAt(walking(bww1gt_1),225). orientation(bww1gt_2,167,225). coords(bww1gt_2,35,166,225).  visible(bww1gt_2,225). happensAt(walking(bww1gt_2),225).
orientation(bww1gt_1,174,226). coords(bww1gt_1,93,262,226).  visible(bww1gt_1,226). happensAt(walking(bww1gt_1),226). orientation(bww1gt_2,167,226). coords(bww1gt_2,37,166,226).  visible(bww1gt_2,226). happensAt(walking(bww1gt_2),226).
orientation(bww1gt_1,173,227). coords(bww1gt_1,92,260,227).  visible(bww1gt_1,227). happensAt(walking(bww1gt_1),227). orientation(bww1gt_2,167,227). coords(bww1gt_2,37,166,227).  visible(bww1gt_2,227). happensAt(walking(bww1gt_2),227).
orientation(bww1gt_1,178,228). coords(bww1gt_1,91,260,228).  visible(bww1gt_1,228). happensAt(walking(bww1gt_1),228). orientation(bww1gt_2,167,228). coords(bww1gt_2,38,166,228).  visible(bww1gt_2,228). happensAt(walking(bww1gt_2),228).
orientation(bww1gt_1,174,229). coords(bww1gt_1,89,259,229).  visible(bww1gt_1,229). happensAt(walking(bww1gt_1),229). orientation(bww1gt_2,167,229). coords(bww1gt_2,38,166,229).  visible(bww1gt_2,229). happensAt(walking(bww1gt_2),229).
orientation(bww1gt_1,170,230). coords(bww1gt_1,88,259,230).  visible(bww1gt_1,230). happensAt(walking(bww1gt_1),230). orientation(bww1gt_2,167,230). coords(bww1gt_2,38,166,230).  visible(bww1gt_2,230). happensAt(walking(bww1gt_2),230).
orientation(bww1gt_1,172,231). coords(bww1gt_1,87,258,231).  visible(bww1gt_1,231). happensAt(walking(bww1gt_1),231). orientation(bww1gt_2,167,231). coords(bww1gt_2,38,166,231).  visible(bww1gt_2,231). happensAt(walking(bww1gt_2),231).
orientation(bww1gt_1,172,232). coords(bww1gt_1,87,257,232).  visible(bww1gt_1,232). happensAt(walking(bww1gt_1),232). orientation(bww1gt_2,167,232). coords(bww1gt_2,39,166,232).  visible(bww1gt_2,232). happensAt(walking(bww1gt_2),232).
orientation(bww1gt_1,176,233). coords(bww1gt_1,86,257,233).  visible(bww1gt_1,233). happensAt(walking(bww1gt_1),233). orientation(bww1gt_2,167,233). coords(bww1gt_2,39,166,233).  visible(bww1gt_2,233). happensAt(walking(bww1gt_2),233).
orientation(bww1gt_1,174,234). coords(bww1gt_1,86,257,234).  visible(bww1gt_1,234). happensAt(walking(bww1gt_1),234). orientation(bww1gt_2,167,234). coords(bww1gt_2,39,166,234).  visible(bww1gt_2,234). happensAt(walking(bww1gt_2),234).
orientation(bww1gt_1,170,235). coords(bww1gt_1,86,257,235).  visible(bww1gt_1,235). happensAt(walking(bww1gt_1),235). orientation(bww1gt_2,167,235). coords(bww1gt_2,39,165,235).  visible(bww1gt_2,235). happensAt(walking(bww1gt_2),235).
orientation(bww1gt_1,169,236). coords(bww1gt_1,86,257,236).  visible(bww1gt_1,236). happensAt(walking(bww1gt_1),236). orientation(bww1gt_2,167,236). coords(bww1gt_2,39,165,236).  visible(bww1gt_2,236). happensAt(walking(bww1gt_2),236).
orientation(bww1gt_1,173,237). coords(bww1gt_1,86,257,237).  visible(bww1gt_1,237). happensAt(walking(bww1gt_1),237). orientation(bww1gt_2,167,237). coords(bww1gt_2,39,165,237).  visible(bww1gt_2,237). happensAt(walking(bww1gt_2),237).
orientation(bww1gt_1,172,238). coords(bww1gt_1,86,257,238).  visible(bww1gt_1,238). happensAt(walking(bww1gt_1),238). orientation(bww1gt_2,167,238). coords(bww1gt_2,39,165,238).  visible(bww1gt_2,238). happensAt(walking(bww1gt_2),238).
orientation(bww1gt_1,174,239). coords(bww1gt_1,86,257,239).  visible(bww1gt_1,239). happensAt(walking(bww1gt_1),239). orientation(bww1gt_2,167,239). coords(bww1gt_2,39,165,239).  visible(bww1gt_2,239). happensAt(walking(bww1gt_2),239).
orientation(bww1gt_1,177,240). coords(bww1gt_1,85,257,240).  visible(bww1gt_1,240). happensAt(walking(bww1gt_1),240). orientation(bww1gt_2,167,240). coords(bww1gt_2,39,165,240).  visible(bww1gt_2,240). happensAt(walking(bww1gt_2),240).
orientation(bww1gt_1,5,241). coords(bww1gt_1,85,256,241).  visible(bww1gt_1,241). happensAt(walking(bww1gt_1),241). orientation(bww1gt_2,167,241). coords(bww1gt_2,40,165,241).  visible(bww1gt_2,241). happensAt(walking(bww1gt_2),241).
orientation(bww1gt_1,7,242). coords(bww1gt_1,85,256,242).  visible(bww1gt_1,242). happensAt(walking(bww1gt_1),242). orientation(bww1gt_2,167,242). coords(bww1gt_2,40,165,242).  visible(bww1gt_2,242). happensAt(walking(bww1gt_2),242).
#show holdsAt/2.