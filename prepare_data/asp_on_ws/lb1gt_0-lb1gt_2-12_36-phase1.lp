
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

	orientation(lb1gt_0,155,12). coords(lb1gt_0,175,172,12).  visible(lb1gt_0,12). happensAt(walking(lb1gt_0),12). orientation(lb1gt_2,142,12). coords(lb1gt_2,80,66,12).  visible(lb1gt_2,12). happensAt(walking(lb1gt_2),12).
orientation(lb1gt_0,155,13). coords(lb1gt_0,174,168,13).  visible(lb1gt_0,13). happensAt(walking(lb1gt_0),13). orientation(lb1gt_2,142,13). coords(lb1gt_2,80,66,13).  visible(lb1gt_2,13). happensAt(walking(lb1gt_2),13).
orientation(lb1gt_0,155,14). coords(lb1gt_0,174,167,14).  visible(lb1gt_0,14). happensAt(walking(lb1gt_0),14). orientation(lb1gt_2,142,14). coords(lb1gt_2,80,66,14).  visible(lb1gt_2,14). happensAt(walking(lb1gt_2),14).
orientation(lb1gt_0,155,15). coords(lb1gt_0,174,165,15).  visible(lb1gt_0,15). happensAt(walking(lb1gt_0),15). orientation(lb1gt_2,142,15). coords(lb1gt_2,80,66,15).  visible(lb1gt_2,15). happensAt(walking(lb1gt_2),15).
orientation(lb1gt_0,150,16). coords(lb1gt_0,174,164,16).  visible(lb1gt_0,16). happensAt(walking(lb1gt_0),16). orientation(lb1gt_2,142,16). coords(lb1gt_2,80,66,16).  visible(lb1gt_2,16). happensAt(walking(lb1gt_2),16).
orientation(lb1gt_0,146,17). coords(lb1gt_0,174,164,17).  visible(lb1gt_0,17). happensAt(walking(lb1gt_0),17). orientation(lb1gt_2,142,17). coords(lb1gt_2,80,67,17).  visible(lb1gt_2,17). happensAt(walking(lb1gt_2),17).
orientation(lb1gt_0,144,18). coords(lb1gt_0,173,161,18).  visible(lb1gt_0,18). happensAt(walking(lb1gt_0),18). orientation(lb1gt_2,142,18). coords(lb1gt_2,80,67,18).  visible(lb1gt_2,18). happensAt(walking(lb1gt_2),18).
orientation(lb1gt_0,146,19). coords(lb1gt_0,173,160,19).  visible(lb1gt_0,19). happensAt(walking(lb1gt_0),19). orientation(lb1gt_2,142,19). coords(lb1gt_2,80,67,19).  visible(lb1gt_2,19). happensAt(walking(lb1gt_2),19).
orientation(lb1gt_0,146,20). coords(lb1gt_0,172,156,20).  visible(lb1gt_0,20). happensAt(walking(lb1gt_0),20). orientation(lb1gt_2,142,20). coords(lb1gt_2,80,68,20).  visible(lb1gt_2,20). happensAt(walking(lb1gt_2),20).
orientation(lb1gt_0,146,21). coords(lb1gt_0,171,153,21).  visible(lb1gt_0,21). happensAt(walking(lb1gt_0),21). orientation(lb1gt_2,142,21). coords(lb1gt_2,80,69,21).  visible(lb1gt_2,21). happensAt(walking(lb1gt_2),21).
orientation(lb1gt_0,146,22). coords(lb1gt_0,171,150,22).  visible(lb1gt_0,22). happensAt(walking(lb1gt_0),22). orientation(lb1gt_2,142,22). coords(lb1gt_2,80,70,22).  visible(lb1gt_2,22). happensAt(walking(lb1gt_2),22).
orientation(lb1gt_0,146,23). coords(lb1gt_0,171,149,23).  visible(lb1gt_0,23). happensAt(walking(lb1gt_0),23). orientation(lb1gt_2,142,23). coords(lb1gt_2,80,71,23).  visible(lb1gt_2,23). happensAt(walking(lb1gt_2),23).
orientation(lb1gt_0,146,24). coords(lb1gt_0,171,148,24).  visible(lb1gt_0,24). happensAt(walking(lb1gt_0),24). orientation(lb1gt_2,142,24). coords(lb1gt_2,80,71,24).  visible(lb1gt_2,24). happensAt(walking(lb1gt_2),24).
orientation(lb1gt_0,146,25). coords(lb1gt_0,171,147,25).  visible(lb1gt_0,25). happensAt(walking(lb1gt_0),25). orientation(lb1gt_2,142,25). coords(lb1gt_2,80,72,25).  visible(lb1gt_2,25). happensAt(walking(lb1gt_2),25).
orientation(lb1gt_0,146,26). coords(lb1gt_0,172,146,26).  visible(lb1gt_0,26). happensAt(walking(lb1gt_0),26). orientation(lb1gt_2,142,26). coords(lb1gt_2,80,72,26).  visible(lb1gt_2,26). happensAt(walking(lb1gt_2),26).
orientation(lb1gt_0,146,27). coords(lb1gt_0,172,144,27).  visible(lb1gt_0,27). happensAt(walking(lb1gt_0),27). orientation(lb1gt_2,142,27). coords(lb1gt_2,80,73,27).  visible(lb1gt_2,27). happensAt(walking(lb1gt_2),27).
orientation(lb1gt_0,146,28). coords(lb1gt_0,173,143,28).  visible(lb1gt_0,28). happensAt(walking(lb1gt_0),28). orientation(lb1gt_2,142,28). coords(lb1gt_2,79,73,28).  visible(lb1gt_2,28). happensAt(walking(lb1gt_2),28).
orientation(lb1gt_0,146,29). coords(lb1gt_0,173,141,29).  visible(lb1gt_0,29). happensAt(walking(lb1gt_0),29). orientation(lb1gt_2,142,29). coords(lb1gt_2,79,74,29).  visible(lb1gt_2,29). happensAt(walking(lb1gt_2),29).
orientation(lb1gt_0,146,30). coords(lb1gt_0,174,138,30).  visible(lb1gt_0,30). happensAt(walking(lb1gt_0),30). orientation(lb1gt_2,142,30). coords(lb1gt_2,79,74,30).  visible(lb1gt_2,30). happensAt(walking(lb1gt_2),30).
orientation(lb1gt_0,146,31). coords(lb1gt_0,174,137,31).  visible(lb1gt_0,31). happensAt(walking(lb1gt_0),31). orientation(lb1gt_2,142,31). coords(lb1gt_2,79,74,31).  visible(lb1gt_2,31). happensAt(walking(lb1gt_2),31).
orientation(lb1gt_0,146,32). coords(lb1gt_0,175,134,32).  visible(lb1gt_0,32). happensAt(walking(lb1gt_0),32). orientation(lb1gt_2,142,32). coords(lb1gt_2,79,74,32).  visible(lb1gt_2,32). happensAt(walking(lb1gt_2),32).
orientation(lb1gt_0,149,33). coords(lb1gt_0,175,132,33).  visible(lb1gt_0,33). happensAt(walking(lb1gt_0),33). orientation(lb1gt_2,142,33). coords(lb1gt_2,79,74,33).  visible(lb1gt_2,33). happensAt(walking(lb1gt_2),33).
orientation(lb1gt_0,157,34). coords(lb1gt_0,175,130,34).  visible(lb1gt_0,34). happensAt(walking(lb1gt_0),34). orientation(lb1gt_2,142,34). coords(lb1gt_2,80,75,34).  visible(lb1gt_2,34). happensAt(walking(lb1gt_2),34).
orientation(lb1gt_0,155,35). coords(lb1gt_0,177,129,35).  visible(lb1gt_0,35). happensAt(walking(lb1gt_0),35). orientation(lb1gt_2,142,35). coords(lb1gt_2,80,75,35).  visible(lb1gt_2,35). happensAt(walking(lb1gt_2),35).
orientation(lb1gt_0,156,36). coords(lb1gt_0,177,129,36).  visible(lb1gt_0,36). happensAt(walking(lb1gt_0),36). orientation(lb1gt_2,142,36). coords(lb1gt_2,80,75,36).  visible(lb1gt_2,36). happensAt(walking(lb1gt_2),36).
#show holdsAt/2.