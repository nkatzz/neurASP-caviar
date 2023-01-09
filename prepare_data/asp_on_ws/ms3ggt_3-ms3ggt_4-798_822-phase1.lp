
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

	orientation(ms3ggt_3,157,798). coords(ms3ggt_3,45,107,798).  visible(ms3ggt_3,798). happensAt(walking(ms3ggt_3),798). orientation(ms3ggt_4,120,798). coords(ms3ggt_4,142,28,798).  visible(ms3ggt_4,798). happensAt(walking(ms3ggt_4),798).
orientation(ms3ggt_3,157,799). coords(ms3ggt_3,45,108,799).  visible(ms3ggt_3,799). happensAt(walking(ms3ggt_3),799). orientation(ms3ggt_4,120,799). coords(ms3ggt_4,142,28,799).  visible(ms3ggt_4,799). happensAt(walking(ms3ggt_4),799).
orientation(ms3ggt_3,157,800). coords(ms3ggt_3,45,108,800).  visible(ms3ggt_3,800). happensAt(walking(ms3ggt_3),800). orientation(ms3ggt_4,120,800). coords(ms3ggt_4,142,28,800).  visible(ms3ggt_4,800). happensAt(walking(ms3ggt_4),800).
orientation(ms3ggt_3,147,801). coords(ms3ggt_3,44,110,801).  visible(ms3ggt_3,801). happensAt(walking(ms3ggt_3),801). orientation(ms3ggt_4,120,801). coords(ms3ggt_4,141,29,801).  visible(ms3ggt_4,801). happensAt(walking(ms3ggt_4),801).
orientation(ms3ggt_3,147,802). coords(ms3ggt_3,44,112,802).  visible(ms3ggt_3,802). happensAt(walking(ms3ggt_3),802). orientation(ms3ggt_4,120,802). coords(ms3ggt_4,140,29,802).  visible(ms3ggt_4,802). happensAt(walking(ms3ggt_4),802).
orientation(ms3ggt_3,147,803). coords(ms3ggt_3,44,113,803).  visible(ms3ggt_3,803). happensAt(walking(ms3ggt_3),803). orientation(ms3ggt_4,120,803). coords(ms3ggt_4,139,29,803).  visible(ms3ggt_4,803). happensAt(walking(ms3ggt_4),803).
orientation(ms3ggt_3,147,804). coords(ms3ggt_3,46,115,804).  visible(ms3ggt_3,804). happensAt(walking(ms3ggt_3),804). orientation(ms3ggt_4,120,804). coords(ms3ggt_4,139,29,804).  visible(ms3ggt_4,804). happensAt(walking(ms3ggt_4),804).
orientation(ms3ggt_3,147,805). coords(ms3ggt_3,46,115,805).  visible(ms3ggt_3,805). happensAt(walking(ms3ggt_3),805). orientation(ms3ggt_4,120,805). coords(ms3ggt_4,139,29,805).  visible(ms3ggt_4,805). happensAt(walking(ms3ggt_4),805).
orientation(ms3ggt_3,147,806). coords(ms3ggt_3,46,117,806).  visible(ms3ggt_3,806). happensAt(walking(ms3ggt_3),806). orientation(ms3ggt_4,120,806). coords(ms3ggt_4,139,29,806).  visible(ms3ggt_4,806). happensAt(walking(ms3ggt_4),806).
orientation(ms3ggt_3,147,807). coords(ms3ggt_3,44,119,807).  visible(ms3ggt_3,807). happensAt(walking(ms3ggt_3),807). orientation(ms3ggt_4,120,807). coords(ms3ggt_4,139,30,807).  visible(ms3ggt_4,807). happensAt(walking(ms3ggt_4),807).
orientation(ms3ggt_3,147,808). coords(ms3ggt_3,44,119,808).  visible(ms3ggt_3,808). happensAt(walking(ms3ggt_3),808). orientation(ms3ggt_4,120,808). coords(ms3ggt_4,139,30,808).  visible(ms3ggt_4,808). happensAt(walking(ms3ggt_4),808).
orientation(ms3ggt_3,147,809). coords(ms3ggt_3,45,119,809).  visible(ms3ggt_3,809). happensAt(walking(ms3ggt_3),809). orientation(ms3ggt_4,120,809). coords(ms3ggt_4,139,30,809).  visible(ms3ggt_4,809). happensAt(walking(ms3ggt_4),809).
orientation(ms3ggt_3,147,810). coords(ms3ggt_3,44,121,810).  visible(ms3ggt_3,810). happensAt(walking(ms3ggt_3),810). orientation(ms3ggt_4,120,810). coords(ms3ggt_4,138,30,810).  visible(ms3ggt_4,810). happensAt(walking(ms3ggt_4),810).
orientation(ms3ggt_3,147,811). coords(ms3ggt_3,44,122,811).  visible(ms3ggt_3,811). happensAt(walking(ms3ggt_3),811). orientation(ms3ggt_4,120,811). coords(ms3ggt_4,138,32,811).  visible(ms3ggt_4,811). happensAt(walking(ms3ggt_4),811).
orientation(ms3ggt_3,147,812). coords(ms3ggt_3,44,126,812).  visible(ms3ggt_3,812). happensAt(walking(ms3ggt_3),812). orientation(ms3ggt_4,131,812). coords(ms3ggt_4,138,32,812).  visible(ms3ggt_4,812). happensAt(walking(ms3ggt_4),812).
orientation(ms3ggt_3,147,813). coords(ms3ggt_3,44,126,813).  visible(ms3ggt_3,813). happensAt(walking(ms3ggt_3),813). orientation(ms3ggt_4,131,813). coords(ms3ggt_4,138,32,813).  visible(ms3ggt_4,813). happensAt(walking(ms3ggt_4),813).
orientation(ms3ggt_3,147,814). coords(ms3ggt_3,44,127,814).  visible(ms3ggt_3,814). happensAt(walking(ms3ggt_3),814). orientation(ms3ggt_4,131,814). coords(ms3ggt_4,137,32,814).  visible(ms3ggt_4,814). happensAt(walking(ms3ggt_4),814).
orientation(ms3ggt_3,147,815). coords(ms3ggt_3,44,127,815).  visible(ms3ggt_3,815). happensAt(walking(ms3ggt_3),815). orientation(ms3ggt_4,131,815). coords(ms3ggt_4,137,32,815).  visible(ms3ggt_4,815). happensAt(walking(ms3ggt_4),815).
orientation(ms3ggt_3,147,816). coords(ms3ggt_3,44,128,816).  visible(ms3ggt_3,816). happensAt(walking(ms3ggt_3),816). orientation(ms3ggt_4,131,816). coords(ms3ggt_4,137,32,816).  visible(ms3ggt_4,816). happensAt(walking(ms3ggt_4),816).
orientation(ms3ggt_3,147,817). coords(ms3ggt_3,44,129,817).  visible(ms3ggt_3,817). happensAt(walking(ms3ggt_3),817). orientation(ms3ggt_4,131,817). coords(ms3ggt_4,137,32,817).  visible(ms3ggt_4,817). happensAt(walking(ms3ggt_4),817).
orientation(ms3ggt_3,153,818). coords(ms3ggt_3,44,129,818).  visible(ms3ggt_3,818). happensAt(walking(ms3ggt_3),818). orientation(ms3ggt_4,131,818). coords(ms3ggt_4,136,32,818).  visible(ms3ggt_4,818). happensAt(walking(ms3ggt_4),818).
orientation(ms3ggt_3,153,819). coords(ms3ggt_3,42,131,819).  visible(ms3ggt_3,819). happensAt(walking(ms3ggt_3),819). orientation(ms3ggt_4,131,819). coords(ms3ggt_4,136,32,819).  visible(ms3ggt_4,819). happensAt(walking(ms3ggt_4),819).
orientation(ms3ggt_3,153,820). coords(ms3ggt_3,42,132,820).  visible(ms3ggt_3,820). happensAt(walking(ms3ggt_3),820). orientation(ms3ggt_4,131,820). coords(ms3ggt_4,136,32,820).  visible(ms3ggt_4,820). happensAt(walking(ms3ggt_4),820).
orientation(ms3ggt_3,153,821). coords(ms3ggt_3,42,132,821).  visible(ms3ggt_3,821). happensAt(walking(ms3ggt_3),821). orientation(ms3ggt_4,131,821). coords(ms3ggt_4,136,32,821).  visible(ms3ggt_4,821). happensAt(walking(ms3ggt_4),821).
orientation(ms3ggt_3,153,822). coords(ms3ggt_3,41,133,822).  visible(ms3ggt_3,822). happensAt(walking(ms3ggt_3),822). orientation(ms3ggt_4,131,822). coords(ms3ggt_4,136,33,822).  visible(ms3ggt_4,822). happensAt(walking(ms3ggt_4),822).
#show holdsAt/2.