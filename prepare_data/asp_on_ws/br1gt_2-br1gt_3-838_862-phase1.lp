
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

	orientation(br1gt_2,12,838). coords(br1gt_2,86,209,838).  visible(br1gt_2,838). happensAt(inactive(br1gt_2),838). orientation(br1gt_3,157,838). coords(br1gt_3,278,208,838).  visible(br1gt_3,838). happensAt(walking(br1gt_3),838).
orientation(br1gt_2,12,839). coords(br1gt_2,86,209,839).  visible(br1gt_2,839). happensAt(inactive(br1gt_2),839). orientation(br1gt_3,157,839). coords(br1gt_3,278,208,839).  visible(br1gt_3,839). happensAt(walking(br1gt_3),839).
orientation(br1gt_2,12,840). coords(br1gt_2,86,209,840).  visible(br1gt_2,840). happensAt(inactive(br1gt_2),840). orientation(br1gt_3,157,840). coords(br1gt_3,278,208,840).  visible(br1gt_3,840). happensAt(walking(br1gt_3),840).
orientation(br1gt_2,12,841). coords(br1gt_2,86,209,841).  visible(br1gt_2,841). happensAt(inactive(br1gt_2),841). orientation(br1gt_3,157,841). coords(br1gt_3,278,205,841).  visible(br1gt_3,841). happensAt(walking(br1gt_3),841).
orientation(br1gt_2,12,842). coords(br1gt_2,86,209,842).  visible(br1gt_2,842). happensAt(inactive(br1gt_2),842). orientation(br1gt_3,157,842). coords(br1gt_3,276,204,842).  visible(br1gt_3,842). happensAt(walking(br1gt_3),842).
orientation(br1gt_2,12,843). coords(br1gt_2,86,209,843).  visible(br1gt_2,843). happensAt(inactive(br1gt_2),843). orientation(br1gt_3,157,843). coords(br1gt_3,276,203,843).  visible(br1gt_3,843). happensAt(walking(br1gt_3),843).
orientation(br1gt_2,12,844). coords(br1gt_2,86,209,844).  visible(br1gt_2,844). happensAt(inactive(br1gt_2),844). orientation(br1gt_3,157,844). coords(br1gt_3,277,203,844).  visible(br1gt_3,844). happensAt(walking(br1gt_3),844).
orientation(br1gt_2,12,845). coords(br1gt_2,86,209,845).  visible(br1gt_2,845). happensAt(inactive(br1gt_2),845). orientation(br1gt_3,157,845). coords(br1gt_3,277,203,845).  visible(br1gt_3,845). happensAt(walking(br1gt_3),845).
orientation(br1gt_2,12,846). coords(br1gt_2,86,209,846).  visible(br1gt_2,846). happensAt(inactive(br1gt_2),846). orientation(br1gt_3,157,846). coords(br1gt_3,277,201,846).  visible(br1gt_3,846). happensAt(walking(br1gt_3),846).
orientation(br1gt_2,12,847). coords(br1gt_2,86,209,847).  visible(br1gt_2,847). happensAt(inactive(br1gt_2),847). orientation(br1gt_3,157,847). coords(br1gt_3,277,201,847).  visible(br1gt_3,847). happensAt(walking(br1gt_3),847).
orientation(br1gt_2,12,848). coords(br1gt_2,86,209,848).  visible(br1gt_2,848). happensAt(inactive(br1gt_2),848). orientation(br1gt_3,157,848). coords(br1gt_3,277,199,848).  visible(br1gt_3,848). happensAt(walking(br1gt_3),848).
orientation(br1gt_2,12,849). coords(br1gt_2,86,209,849).  visible(br1gt_2,849). happensAt(inactive(br1gt_2),849). orientation(br1gt_3,157,849). coords(br1gt_3,277,198,849).  visible(br1gt_3,849). happensAt(walking(br1gt_3),849).
orientation(br1gt_2,12,850). coords(br1gt_2,86,209,850).  visible(br1gt_2,850). happensAt(inactive(br1gt_2),850). orientation(br1gt_3,157,850). coords(br1gt_3,278,196,850).  visible(br1gt_3,850). happensAt(walking(br1gt_3),850).
orientation(br1gt_2,12,851). coords(br1gt_2,86,209,851).  visible(br1gt_2,851). happensAt(inactive(br1gt_2),851). orientation(br1gt_3,150,851). coords(br1gt_3,278,194,851).  visible(br1gt_3,851). happensAt(walking(br1gt_3),851).
orientation(br1gt_2,12,852). coords(br1gt_2,86,209,852).  visible(br1gt_2,852). happensAt(inactive(br1gt_2),852). orientation(br1gt_3,150,852). coords(br1gt_3,281,192,852).  visible(br1gt_3,852). happensAt(walking(br1gt_3),852).
orientation(br1gt_2,12,853). coords(br1gt_2,85,208,853).  visible(br1gt_2,853). happensAt(inactive(br1gt_2),853). orientation(br1gt_3,150,853). coords(br1gt_3,282,190,853).  visible(br1gt_3,853). happensAt(walking(br1gt_3),853).
orientation(br1gt_2,12,854). coords(br1gt_2,85,208,854).  visible(br1gt_2,854). happensAt(inactive(br1gt_2),854). orientation(br1gt_3,145,854). coords(br1gt_3,284,188,854).  visible(br1gt_3,854). happensAt(walking(br1gt_3),854).
orientation(br1gt_2,12,855). coords(br1gt_2,85,208,855).  visible(br1gt_2,855). happensAt(inactive(br1gt_2),855). orientation(br1gt_3,145,855). coords(br1gt_3,285,187,855).  visible(br1gt_3,855). happensAt(walking(br1gt_3),855).
orientation(br1gt_2,12,856). coords(br1gt_2,85,208,856).  visible(br1gt_2,856). happensAt(inactive(br1gt_2),856). orientation(br1gt_3,145,856). coords(br1gt_3,287,187,856).  visible(br1gt_3,856). happensAt(walking(br1gt_3),856).
orientation(br1gt_2,12,857). coords(br1gt_2,85,208,857).  visible(br1gt_2,857). happensAt(inactive(br1gt_2),857). orientation(br1gt_3,145,857). coords(br1gt_3,287,187,857).  visible(br1gt_3,857). happensAt(walking(br1gt_3),857).
orientation(br1gt_2,12,858). coords(br1gt_2,85,208,858).  visible(br1gt_2,858). happensAt(inactive(br1gt_2),858). orientation(br1gt_3,145,858). coords(br1gt_3,287,186,858).  visible(br1gt_3,858). happensAt(walking(br1gt_3),858).
orientation(br1gt_2,12,859). coords(br1gt_2,85,208,859).  visible(br1gt_2,859). happensAt(inactive(br1gt_2),859). orientation(br1gt_3,145,859). coords(br1gt_3,287,185,859).  visible(br1gt_3,859). happensAt(walking(br1gt_3),859).
orientation(br1gt_2,12,860). coords(br1gt_2,85,208,860).  visible(br1gt_2,860). happensAt(inactive(br1gt_2),860). orientation(br1gt_3,145,860). coords(br1gt_3,287,185,860).  visible(br1gt_3,860). happensAt(walking(br1gt_3),860).
orientation(br1gt_2,12,861). coords(br1gt_2,85,208,861).  visible(br1gt_2,861). happensAt(inactive(br1gt_2),861). orientation(br1gt_3,154,861). coords(br1gt_3,288,182,861).  visible(br1gt_3,861). happensAt(walking(br1gt_3),861).
orientation(br1gt_2,12,862). coords(br1gt_2,85,208,862).  visible(br1gt_2,862). happensAt(inactive(br1gt_2),862). orientation(br1gt_3,152,862). coords(br1gt_3,289,182,862).  visible(br1gt_3,862). happensAt(walking(br1gt_3),862).
#show holdsAt/2.