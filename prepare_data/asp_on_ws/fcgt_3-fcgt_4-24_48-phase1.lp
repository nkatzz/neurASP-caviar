
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

	orientation(fcgt_3,159,24). coords(fcgt_3,103,163,24).  visible(fcgt_3,24). happensAt(inactive(fcgt_3),24). orientation(fcgt_4,166,24). coords(fcgt_4,103,214,24).  visible(fcgt_4,24). happensAt(walking(fcgt_4),24).
orientation(fcgt_3,159,25). coords(fcgt_3,103,163,25).  visible(fcgt_3,25). happensAt(inactive(fcgt_3),25). orientation(fcgt_4,166,25). coords(fcgt_4,103,214,25).  visible(fcgt_4,25). happensAt(walking(fcgt_4),25).
orientation(fcgt_3,159,26). coords(fcgt_3,103,163,26).  visible(fcgt_3,26). happensAt(inactive(fcgt_3),26). orientation(fcgt_4,166,26). coords(fcgt_4,102,214,26).  visible(fcgt_4,26). happensAt(walking(fcgt_4),26).
orientation(fcgt_3,159,27). coords(fcgt_3,103,163,27).  visible(fcgt_3,27). happensAt(inactive(fcgt_3),27). orientation(fcgt_4,166,27). coords(fcgt_4,101,214,27).  visible(fcgt_4,27). happensAt(walking(fcgt_4),27).
orientation(fcgt_3,159,28). coords(fcgt_3,103,163,28).  visible(fcgt_3,28). happensAt(inactive(fcgt_3),28). orientation(fcgt_4,166,28). coords(fcgt_4,101,212,28).  visible(fcgt_4,28). happensAt(walking(fcgt_4),28).
orientation(fcgt_3,159,29). coords(fcgt_3,103,163,29).  visible(fcgt_3,29). happensAt(inactive(fcgt_3),29). orientation(fcgt_4,166,29). coords(fcgt_4,101,212,29).  visible(fcgt_4,29). happensAt(walking(fcgt_4),29).
orientation(fcgt_3,159,30). coords(fcgt_3,103,163,30).  visible(fcgt_3,30). happensAt(inactive(fcgt_3),30). orientation(fcgt_4,166,30). coords(fcgt_4,101,212,30).  visible(fcgt_4,30). happensAt(walking(fcgt_4),30).
orientation(fcgt_3,159,31). coords(fcgt_3,103,163,31).  visible(fcgt_3,31). happensAt(inactive(fcgt_3),31). orientation(fcgt_4,166,31). coords(fcgt_4,101,212,31).  visible(fcgt_4,31). happensAt(walking(fcgt_4),31).
orientation(fcgt_3,159,32). coords(fcgt_3,103,163,32).  visible(fcgt_3,32). happensAt(inactive(fcgt_3),32). orientation(fcgt_4,166,32). coords(fcgt_4,101,212,32).  visible(fcgt_4,32). happensAt(walking(fcgt_4),32).
orientation(fcgt_3,159,33). coords(fcgt_3,103,163,33).  visible(fcgt_3,33). happensAt(inactive(fcgt_3),33). orientation(fcgt_4,166,33). coords(fcgt_4,101,212,33).  visible(fcgt_4,33). happensAt(walking(fcgt_4),33).
orientation(fcgt_3,159,34). coords(fcgt_3,103,163,34).  visible(fcgt_3,34). happensAt(inactive(fcgt_3),34). orientation(fcgt_4,166,34). coords(fcgt_4,101,212,34).  visible(fcgt_4,34). happensAt(walking(fcgt_4),34).
orientation(fcgt_3,159,35). coords(fcgt_3,103,163,35).  visible(fcgt_3,35). happensAt(inactive(fcgt_3),35). orientation(fcgt_4,166,35). coords(fcgt_4,101,212,35).  visible(fcgt_4,35). happensAt(walking(fcgt_4),35).
orientation(fcgt_3,159,36). coords(fcgt_3,103,163,36).  visible(fcgt_3,36). happensAt(inactive(fcgt_3),36). orientation(fcgt_4,169,36). coords(fcgt_4,101,212,36).  visible(fcgt_4,36). happensAt(walking(fcgt_4),36).
orientation(fcgt_3,159,37). coords(fcgt_3,103,163,37).  visible(fcgt_3,37). happensAt(inactive(fcgt_3),37). orientation(fcgt_4,169,37). coords(fcgt_4,101,212,37).  visible(fcgt_4,37). happensAt(walking(fcgt_4),37).
orientation(fcgt_3,159,38). coords(fcgt_3,103,163,38).  visible(fcgt_3,38). happensAt(inactive(fcgt_3),38). orientation(fcgt_4,169,38). coords(fcgt_4,101,212,38).  visible(fcgt_4,38). happensAt(walking(fcgt_4),38).
orientation(fcgt_3,159,39). coords(fcgt_3,103,163,39).  visible(fcgt_3,39). happensAt(inactive(fcgt_3),39). orientation(fcgt_4,169,39). coords(fcgt_4,101,212,39).  visible(fcgt_4,39). happensAt(walking(fcgt_4),39).
orientation(fcgt_3,159,40). coords(fcgt_3,103,163,40).  visible(fcgt_3,40). happensAt(inactive(fcgt_3),40). orientation(fcgt_4,169,40). coords(fcgt_4,101,212,40).  visible(fcgt_4,40). happensAt(walking(fcgt_4),40).
orientation(fcgt_3,159,41). coords(fcgt_3,103,163,41).  visible(fcgt_3,41). happensAt(inactive(fcgt_3),41). orientation(fcgt_4,169,41). coords(fcgt_4,101,211,41).  visible(fcgt_4,41). happensAt(walking(fcgt_4),41).
orientation(fcgt_3,159,42). coords(fcgt_3,103,163,42).  visible(fcgt_3,42). happensAt(inactive(fcgt_3),42). orientation(fcgt_4,169,42). coords(fcgt_4,101,210,42).  visible(fcgt_4,42). happensAt(walking(fcgt_4),42).
orientation(fcgt_3,159,43). coords(fcgt_3,103,163,43).  visible(fcgt_3,43). happensAt(inactive(fcgt_3),43). orientation(fcgt_4,169,43). coords(fcgt_4,101,210,43).  visible(fcgt_4,43). happensAt(walking(fcgt_4),43).
orientation(fcgt_3,159,44). coords(fcgt_3,103,163,44).  visible(fcgt_3,44). happensAt(inactive(fcgt_3),44). orientation(fcgt_4,169,44). coords(fcgt_4,100,210,44).  visible(fcgt_4,44). happensAt(walking(fcgt_4),44).
orientation(fcgt_3,159,45). coords(fcgt_3,103,163,45).  visible(fcgt_3,45). happensAt(inactive(fcgt_3),45). orientation(fcgt_4,169,45). coords(fcgt_4,100,210,45).  visible(fcgt_4,45). happensAt(walking(fcgt_4),45).
orientation(fcgt_3,159,46). coords(fcgt_3,103,163,46).  visible(fcgt_3,46). happensAt(inactive(fcgt_3),46). orientation(fcgt_4,169,46). coords(fcgt_4,100,210,46).  visible(fcgt_4,46). happensAt(walking(fcgt_4),46).
orientation(fcgt_3,159,47). coords(fcgt_3,103,163,47).  visible(fcgt_3,47). happensAt(inactive(fcgt_3),47). orientation(fcgt_4,169,47). coords(fcgt_4,100,210,47).  visible(fcgt_4,47). happensAt(walking(fcgt_4),47).
orientation(fcgt_3,159,48). coords(fcgt_3,103,163,48).  visible(fcgt_3,48). happensAt(inactive(fcgt_3),48). orientation(fcgt_4,166,48). coords(fcgt_4,100,210,48).  visible(fcgt_4,48). happensAt(walking(fcgt_4),48).
#show holdsAt/2.